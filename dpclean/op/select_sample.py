import math
import os
import random
import shutil
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import dpdata
import numpy as np
from dflow.python import OP, OPIO, Artifact, OPIOSign, Parameter


class Validate(OP, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "valid_systems": Artifact(List[Path]),
                "model": Artifact(Path),
                "train_params": dict,
                "batch_size": Parameter(str, default="auto"),
                "optional_args": Parameter(dict, default={}),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "results": dict,
            }
        )

    def load_model(self, model: Path):
        pass

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def validate(self, systems, train_params, batch_size, optional_args=None):
        metrics = {
            "mae_e": [],
            "rmse_e": [],
            "mae_epa": [],
            "rmse_epa": [],
            "mae_f": [],
            "rmse_f": [],
            "mae_v": [],
            "rmse_v": [],
            "mae_vpa": [],
            "rmse_vpa": [],
            "natoms": [],
        }
        for sys in systems:
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            d = dpdata.MultiSystems()
            if mixed_type:
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                d.append(k)
            for k in d:
                metrics_sys = {key: [] for key in metrics}
                for i in range(len(k)):
                    cell = k.data["cells"][i]
                    if k.nopbc:
                        cell = None
                    coord = k.data["coords"][i]
                    energy0 = k.data["energies"][i] if "energies" in k.data else None
                    force0 = k.data["forces"][i] if "forces" in k.data else None
                    virial0 = k.data["virials"][i] if "virials" in k.data else None
                    ori_atype = k.data["atom_types"]
                    anames = k.data["atom_names"]
                    atype = np.array([train_params["model"]["type_map"].index(anames[j]) for j in ori_atype])
                    n = k.get_natoms()
                    e, f, v = self.evaluate(coord, cell, atype)

                    if energy0 is not None:
                        metrics_sys["mae_e"].append(np.mean(np.abs(e - energy0)))
                        metrics_sys["rmse_e"].append(np.sqrt(np.mean((e - energy0)**2)))
                        metrics_sys["mae_epa"].append(np.mean(np.abs(e - energy0)) / n)
                        metrics_sys["rmse_epa"].append(np.sqrt(np.mean((e - energy0)**2)) / n)
                    if force0 is not None:
                        metrics_sys["mae_f"].append(np.mean(np.abs(f - force0)))
                        metrics_sys["rmse_f"].append(np.sqrt(np.mean((f - force0)**2)))
                    if virial0 is not None:
                        metrics_sys["mae_v"].append(np.mean(np.abs(v - virial0)))
                        metrics_sys["rmse_v"].append(np.sqrt(np.mean((v - virial0)**2)))
                        metrics_sys["mae_vpa"].append(np.mean(np.abs(v - virial0)) / n)
                        metrics_sys["rmse_vpa"].append(np.sqrt(np.mean((v - virial0)**2)) / n)
                    metrics_sys["natoms"].append(n)
                print("System: %s metrics: %s" % (sys, metrics_sys))
                for key in metrics:
                    if len(metrics_sys[key]) > 0:
                        metrics[key].append(metrics_sys[key])
        return {key: np.array(value) for key, value in metrics.items() if len(value) > 0}

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        metrics = self.validate(ip["valid_systems"], ip["train_params"], batch_size=ip["batch_size"], optional_args=ip["optional_args"])
        metrics = {key: np.concatenate(value) for key, value in metrics.items()}
        results = {}
        for key, value in metrics.items():
            if key == "natoms":
                continue
            if key == "mae_f":
                results[key] = float(np.sum(value*metrics["natoms"])/np.sum(metrics["natoms"]))
            elif key == "rmse_f":
                results[key] = float(np.sqrt(np.sum(value**2*metrics["natoms"])/np.sum(metrics["natoms"])))
            elif key.startswith("mae_"):
                results[key] = float(np.mean(value))
            elif key.startswith("rmse_"):
                results[key] = float(np.sqrt(np.mean(value**2)))
        return OPIO({
            "results": results,
        })


class SelectSamples(Validate, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "current_systems": Artifact(List[Path]),
                "candidate_systems": Artifact(List[Path]),
                "valid_systems": Artifact(List[Path]),
                "model": Artifact(Path),
                "iter": int,
                "learning_curve": dict,
                "select_type": Parameter(str, default="global"),
                "ratio_selected": List[float],
                "train_params": dict,
                "batch_size": Parameter(str, default="auto"),
                "optional_args": Parameter(dict, default={}),
                "criteria_metrics": Parameter(str, default="rmse_f"),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "remaining_systems": Artifact(List[Path]),
                "current_systems": Artifact(List[Path]),
                "n_selected": int,
                "n_remaining": int,
                "learning_curve": dict,
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        metrics = self.validate(ip["valid_systems"], ip["train_params"], batch_size=ip["batch_size"], optional_args=ip["optional_args"])
        metrics = {key: np.concatenate(value) for key, value in metrics.items()}
        lcurve = ip["learning_curve"]
        for key, value in metrics.items():
            if key == "natoms":
                continue
            lcurve[key] = lcurve.get(key, [])
            if key == "mae_f":
                lcurve[key].append(float(np.sum(value*metrics["natoms"])/np.sum(metrics["natoms"])))
            elif key == "rmse_f":
                lcurve[key].append(float(np.sqrt(np.sum(value**2*metrics["natoms"])/np.sum(metrics["natoms"]))))
            elif key.startswith("mae_"):
                lcurve[key].append(float(np.mean(value)))
            elif key.startswith("rmse_"):
                lcurve[key].append(float(np.sqrt(np.mean(value**2))))

        n_current = 0
        for sys in ip["current_systems"]:
            k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            n_current += len(k)
        lcurve["nsamples"] = lcurve.get("nsamples", [])
        lcurve["nsamples"].append(n_current)

        ratio_selected = ip["ratio_selected"][ip["iter"]] if ip["iter"] < len(ip["ratio_selected"]) else 0
        if ratio_selected == 0:
            n_ramain = 0
            for sys in ip["candidate_systems"]:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                n_ramain += len(k)
            return OPIO({
                "remaining_systems": ip["candidate_systems"],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": n_ramain,
                "learning_curve": lcurve,
            })

        metrics = self.validate(ip["candidate_systems"], ip["train_params"], batch_size=ip["batch_size"], optional_args=ip["optional_args"])
        criteria = metrics[ip["criteria_metrics"]]
        nf = sum([len(i) for i in criteria])
        if nf == 0:
            return OPIO({
                "remaining_systems": [],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": 0,
                "learning_curve": lcurve,
            })

        indices = [[] for _ in range(len(criteria))]
        if ip["select_type"] == "global":
            mapping = sum([[(i, j) for j in range(len(s))] for i, s in enumerate(criteria)], [])
            criteria_1d = np.concatenate(criteria)
            sorted_indices = [int(i) for i in np.argsort(criteria_1d)]
            sorted_indices.reverse()
            print("Sorted criteria metrics: ", criteria_1d[sorted_indices])
            ns = math.floor(len(sorted_indices) * ratio_selected)
            if random.random() < len(sorted_indices) * ratio_selected - ns:
                ns += 1
            print("Selected criteria metrics: ", criteria_1d[sorted_indices[:ns]])
            for i in range(ns):
                indices[mapping[sorted_indices[i]][0]].append(mapping[sorted_indices[i]][1])
        elif ip["select_type"] == "system":
            for i in range(len(criteria)):
                print("System: ", i)
                if len(criteria[i]) > 0:
                    ns = math.floor(len(criteria[i]) * ratio_selected)
                    if random.random() < len(criteria[i]) * ratio_selected - ns:
                        ns += 1
                    sorted_indices = [int(j) for j in np.argsort(criteria[i])]
                    sorted_indices.reverse()
                    print("Sorted criteria metrics: ", criteria[i][sorted_indices])
                    print("Selected criteria metrics: ", criteria[i][sorted_indices[:ns]])
                    for j in range(ns):
                        indices[i].append(sorted_indices[j])
        n_selected = sum([len(i) for i in indices])

        current_systems = ip["current_systems"]
        remaining_systems = []
        i = 0
        for path in ip["candidate_systems"]:
            d = dpdata.MultiSystems()
            mixed_type = len(list(path.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d.load_systems_from_file(path, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                d.append(k)

            selected_systems = dpdata.MultiSystems()
            unselected_systems = dpdata.MultiSystems()
            for k in d:
                selected_indices = indices[i]
                unselected_indices = list(set(range(len(k))).difference(indices[i]))
                i += 1
                if len(selected_indices) > 0:
                    selected_systems.append(k.sub_system(selected_indices))
                if len(unselected_indices) > 0:
                    unselected_systems.append(k.sub_system(unselected_indices))

            if len(selected_systems) > 0:
                target = Path("iter-%s" % ip["iter"]) / path.relative_to(ip["candidate_systems"].art_root)
                if len(selected_systems) == 1:
                    if mixed_type:
                        selected_systems[0].to_deepmd_npy_mixed(target)
                    else:
                        selected_systems[0].to_deepmd_npy(target)
                else:
                    # The multisystem is loaded from one dir, thus we can safely keep one dir
                    selected_systems.to_deepmd_npy_mixed("%s.tmp" % target)
                    fs = os.listdir("%s.tmp" % target)
                    assert len(fs) == 1
                    os.rename(os.path.join("%s.tmp" % target, fs[0]), target)
                    os.rmdir("%s.tmp" % target)
                current_systems.append(target)

            if len(unselected_systems) > 0:
                target = path
                shutil.rmtree(target)
                if len(unselected_systems) == 1:
                    if mixed_type:
                        unselected_systems[0].to_deepmd_npy_mixed(target)
                    else:
                        unselected_systems[0].to_deepmd_npy(target)
                else:
                    # The multisystem is loaded from one dir, thus we can safely keep one dir
                    unselected_systems.to_deepmd_npy_mixed("%s.tmp" % target)
                    fs = os.listdir("%s.tmp" % target)
                    assert len(fs) == 1
                    os.rename(os.path.join("%s.tmp" % target, fs[0]), target)
                    os.rmdir("%s.tmp" % target)
                remaining_systems.append(target)

        return OPIO({
                "remaining_systems": remaining_systems,
                "current_systems": current_systems,
                "n_selected": n_selected,
                "n_remaining": nf - n_selected,
                "learning_curve": lcurve,
            })
