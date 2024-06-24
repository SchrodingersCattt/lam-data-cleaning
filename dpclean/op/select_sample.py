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
        rmse_f = []
        rmse_e = []
        rmse_v = []
        natoms = []
        for sys in systems:
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            d = dpdata.MultiSystems()
            if mixed_type:
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                d.append(k)
            for k in d:
                rmse_f_sys = []
                rmse_e_sys = []
                rmse_v_sys = []
                natoms_sys = []
                for i in range(len(k)):
                    cell = k[i].data["cells"][0]
                    if k[i].nopbc:
                        cell = None
                    coord = k[i].data["coords"][0]
                    force0 = k[i].data["forces"][0]
                    energy0 = k[i].data["energies"][0]
                    virial0 = k[i].data["virials"][0] if "virials" in k[i].data else None
                    ori_atype = k[i].data["atom_types"]
                    anames = k[i].data["atom_names"]
                    atype = np.array([train_params["model"]["type_map"].index(anames[j]) for j in ori_atype])
                    e, f, v = self.evaluate(coord, cell, atype)

                    lx = 0
                    for j in range(force0.shape[0]):
                        lx += (force0[j][0] - f[j][0]) ** 2 + \
                                (force0[j][1] - f[j][1]) ** 2 + \
                                (force0[j][2] - f[j][2]) ** 2
                    err_f = ( lx / force0.shape[0] / 3 ) ** 0.5
                    err_e = abs(energy0 - e) / force0.shape[0]
                    err_v = np.sqrt(np.average((virial0 - v)**2)) / force0.shape[0] if virial0 is not None else None
                    print("System: %s frame: %s rmse_e: %s rmse_f: %s rmse_v: %s" % (sys, i, err_e, err_f, err_v))
                    rmse_f_sys.append(err_f)
                    rmse_e_sys.append(err_e)
                    if err_v is not None:
                        rmse_v_sys.append(err_v)
                    natoms_sys.append(force0.shape[0])
                rmse_f.append(rmse_f_sys)
                rmse_e.append(rmse_e_sys)
                if len(rmse_v_sys) > 0:
                    rmse_v.append(rmse_v_sys)
                natoms.append(natoms_sys)
        return rmse_f, rmse_e, rmse_v if len(rmse_v) > 0 else None, natoms

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        rmse_f, rmse_e, rmse_v, natoms = self.validate(ip["valid_systems"], ip["train_params"], batch_size=ip["batch_size"], optional_args=ip["optional_args"])
        na = sum([sum(i) for i in natoms])
        nf = sum([len(i) for i in natoms])
        rmse_f = np.sqrt(sum([sum([i**2*j for i, j in zip(r, n)]) for r, n in zip(rmse_f, natoms)]) / na)
        rmse_e = np.sqrt(sum([sum([i**2 for i in r]) for r in rmse_e]) / nf)
        rmse_v = float(np.sqrt(np.average(np.concatenate(rmse_v)**2))) if rmse_v is not None else None
        results = {"rmse_f": float(rmse_f), "rmse_e": float(rmse_e), "rmse_v": rmse_v}
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
        rmse_f, rmse_e, rmse_v, natoms = self.validate(ip["valid_systems"], ip["train_params"], batch_size=ip["batch_size"], optional_args=ip["optional_args"])
        na = sum([sum(i) for i in natoms])
        nf = sum([len(i) for i in natoms])
        rmse_f = np.sqrt(sum([sum([i**2*j for i, j in zip(r, n)]) for r, n in zip(rmse_f, natoms)]) / na)
        rmse_e = np.sqrt(sum([sum([i**2 for i in r]) for r in rmse_e]) / nf)
        rmse_v = float(np.sqrt(np.average(np.concatenate(rmse_v)**2))) if rmse_v is not None else None
        n_current = 0
        for sys in ip["current_systems"]:
            k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            n_current += len(k)
        lcurve = ip["learning_curve"]
        lcurve["nsamples"] = lcurve.get("nsamples", [])
        lcurve["nsamples"].append(n_current)
        lcurve["rmse_f"] = lcurve.get("rmse_f", [])
        lcurve["rmse_f"].append(float(rmse_f))
        lcurve["rmse_e"] = lcurve.get("rmse_e", [])
        lcurve["rmse_e"].append(float(rmse_e))
        if rmse_v is not None:
            lcurve["rmse_v"] = lcurve.get("rmse_v", [])
            lcurve["rmse_v"].append(rmse_v)

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

        rmse_f, _, _, _ = self.validate(ip["candidate_systems"], ip["train_params"], batch_size=ip["batch_size"], optional_args=ip["optional_args"])
        nf = sum([len(i) for i in rmse_f])
        if nf == 0:
            return OPIO({
                "remaining_systems": [],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": 0,
                "learning_curve": lcurve,
            })

        indices = [[] for _ in range(len(rmse_f))]
        if ip["select_type"] == "global":
            mapping = sum([[(i, j) for j in range(len(s))] for i, s in enumerate(rmse_f)], [])
            rmse_f_1d = sum(rmse_f, [])
            sorted_indices = [int(i) for i in np.argsort(rmse_f_1d)]
            sorted_indices.reverse()
            ns = math.floor(len(sorted_indices) * ratio_selected)
            if random.random() < len(sorted_indices) * ratio_selected - ns:
                ns += 1
            for i in range(ns):
                indices[mapping[sorted_indices[i]][0]].append(mapping[sorted_indices[i]][1])
        elif ip["select_type"] == "system":
            for i in range(len(rmse_f)):
                if len(rmse_f[i]) > 0:
                    ns = math.floor(len(rmse_f[i]) * ratio_selected)
                    if random.random() < len(rmse_f[i]) * ratio_selected - ns:
                        ns += 1
                    sorted_indices = [int(j) for j in np.argsort(rmse_f[i])]
                    sorted_indices.reverse()
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
