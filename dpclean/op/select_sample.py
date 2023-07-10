import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import dpdata
import numpy as np
from dflow.python import OP, OPIO, Artifact, OPIOSign, Parameter

type_map = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]


class Validate(OP, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "valid_systems": Artifact(List[Path]),
                "model": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "results": dict,
            }
        )

    @abstractmethod
    def load_model(self, model: Path):
        pass

    @abstractmethod
    def evaluate(self,
                 coord: np.ndarray,
                 cell: np.ndarray,
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def validate(self, systems):
        rmse_f = []
        rmse_e = []
        natoms = []
        for sys in systems:
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d = dpdata.MultiSystems()
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
                k = d[0]
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            rmse_f_sys = []
            rmse_e_sys = []
            natoms_sys = []
            for i in range(len(k)):
                cell = k[i].data["cells"][0]
                coord = k[i].data["coords"][0]
                force0 = k[i].data["forces"][0]
                energy0 = k[i].data["energies"][0]
                ori_atype = k[i].data["atom_types"]
                anames = k[i].data["atom_names"]
                atype = np.array([type_map.index(anames[i]) for i in ori_atype])
                e, f, v = self.evaluate(coord, cell, atype)

                lx = 0
                for i in range(force0.shape[0]):
                    lx += (force0[i][0] - f[i][0]) ** 2 + \
                            (force0[i][1] - f[i][1]) ** 2 + \
                            (force0[i][2] - f[i][2]) ** 2
                err_f = ( lx / force0.shape[0] / 3 ) ** 0.5
                rmse_f_sys.append(err_f)
                rmse_e_sys.append(abs(energy0-e)/force0.shape[0])
                natoms_sys.append(force0.shape[0])
            rmse_f.append(rmse_f_sys)
            rmse_e.append(rmse_e_sys)
            natoms.append(natoms_sys)
        return rmse_f, rmse_e, natoms

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        rmse_f, rmse_e, natoms = self.validate(ip["valid_systems"])
        na = sum([sum(i) for i in natoms])
        rmse_f = np.sqrt(sum([sum([i**2*j for i, j in zip(r, n)]) for r, n in zip(rmse_f, natoms)]) / na)
        rmse_e = np.sqrt(sum([sum([i**2*j for i, j in zip(r, n)]) for r, n in zip(rmse_e, natoms)]) / na)
        results = {"rmse_f": float(rmse_f), "rmse_e": float(rmse_e)}
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
                "max_selected": Union[int, List[int]],
                "iter": int,
                "threshold": float,
                "learning_curve": dict,
                "select_type": Parameter(str, default="global"),
                "ratio_selected": Union[float, List[float]],
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
                "converged": bool,
                "learning_curve": dict,
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        rmse_f, rmse_e, natoms = self.validate(ip["valid_systems"])
        na = sum([sum(i) for i in natoms])
        rmse_f = np.sqrt(sum([sum([i**2*j for i, j in zip(r, n)]) for r, n in zip(rmse_f, natoms)]) / na)
        rmse_e = np.sqrt(sum([sum([i**2*j for i, j in zip(r, n)]) for r, n in zip(rmse_e, natoms)]) / na)
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

        rmse_f, _, _ = self.validate(ip["candidate_systems"])
        nf = sum([len(i) for i in rmse_f])
        if nf == 0:
            return OPIO({
                "remaining_systems": [],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": 0,
                "converged": False,
                "learning_curve": lcurve,
            })
        f_max = max([max(i) for i in rmse_f if len(i) > 0])
        f_avg = sum([sum(i) for i in rmse_f]) / nf
        f_min = min([min(i) for i in rmse_f if len(i) > 0])
        print('max force (eV/A): ', f_max)
        print('avg force (eV/A): ', f_avg)
        print('min force (eV/A): ', f_min)
        if f_max - f_avg <= ip["threshold"] * f_avg:
            return OPIO({
                "remaining_systems": ip["candidate_systems"],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": nf,
                "converged": True,
                "learning_curve": lcurve,
            })

        if isinstance(ip["max_selected"], list):
            max_selected = ip["max_selected"][ip["iter"]] if ip["iter"] < len(ip["max_selected"]) else ip["max_selected"][-1]
        else:
            max_selected = ip["max_selected"] 

        if isinstance(ip["ratio_selected"], list):
            ratio_selected = ip["ratio_selected"][ip["iter"]] if ip["iter"] < len(ip["ratio_selected"]) else ip["ratio_selected"][-1]
        else:
            ratio_selected = ip["ratio_selected"] 

        n = len(ip["candidate_systems"])
        if ip["select_type"] == "global":
            mapping = sum([[(i, j) for j in range(len(s))] for i, s in enumerate(rmse_f)], [])
            rmse_f_1d = sum(rmse_f, [])
            sorted_indices = [int(i) for i in np.argsort(rmse_f_1d)]
            sorted_indices.reverse()
            indices = [mapping[sorted_indices[i]] for i in range(min(len(sorted_indices), max_selected))]
        elif ip["select_type"] == "system":
            indices = []
            for i in range(n):
                if len(rmse_f[i]) > 0:
                    ns = math.ceil(len(rmse_f[i])*ratio_selected)
                    sorted_indices = [int(j) for j in np.argsort(rmse_f[i])]
                    sorted_indices.reverse()
                    indices.extend([(i, sorted_indices[j]) for j in range(ns)])

        current_systems = ip["current_systems"]
        remaining_systems = []
        for i in range(n):
            selected = [index[1] for index in indices if index[0] == i]
            if len(selected) > 0:
                path = ip["candidate_systems"][i]
                mixed_type = len(list(path.glob("*/real_atom_types.npy"))) > 0
                if mixed_type:
                    d = dpdata.MultiSystems()
                    d.load_systems_from_file(path, fmt="deepmd/npy/mixed")
                    k = d[0]
                else:
                    k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                frames = k.sub_system(selected)
                root = str(path)[:str(path).find("candidate_systems") + 17]
                target = Path("iter-%s" % ip["iter"]) / path.relative_to(root)
                if mixed_type:
                    frames.to_deepmd_npy_mixed(target)
                else:
                    frames.to_deepmd_npy(target)
                current_systems.append(target)
                if len(selected) < len(k):
                    target = path
                    remain = k.sub_system([j for j in range(len(k)) if j not in selected])
                    if mixed_type:
                        remain.to_deepmd_npy_mixed(target)
                    else:
                        remain.to_deepmd_npy(target)
                    remaining_systems.append(target)
            else:
                remaining_systems.append(ip["candidate_systems"][i])

        return OPIO({
                "remaining_systems": remaining_systems,
                "current_systems": current_systems,
                "n_selected": len(indices),
                "n_remaining": nf - len(indices),
                "converged": False,
                "learning_curve": lcurve,
            })
