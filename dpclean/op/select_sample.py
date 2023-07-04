from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import dpdata
import numpy as np
from dflow.python import OP, OPIO, Artifact, OPIOSign


class SelectSamples(OP, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "current_systems": Artifact(List[Path]),
                "candidate_systems": Artifact(List[Path]),
                "valid_systems": Artifact(List[Path]),
                "model": Artifact(Path),
                "max_selected": int,
                "threshold": float,
                "learning_curve": list,
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
                "learning_curve": list,
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
        for sys in systems:
            if sys is None:
                rmse_f.append([])
                continue
            k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            rmse_f_sys = []
            for i in range(len(k)):
                cell = k[i].data["cells"][0]
                coord = k[i].data["coords"][0]
                force0 = k[i].data["forces"][0]
                atype = k[i].data["atom_types"]
                e, f, v = self.evaluate(coord, cell, atype)

                lx = 0
                for i in range(force0.shape[0]):
                    lx += (force0[i][0] - f[i][0]) ** 2 + \
                            (force0[i][1] - f[i][1]) ** 2 + \
                            (force0[i][2] - f[i][2]) ** 2
                err_f = ( lx / force0.shape[0] / 3 ) ** 0.5
                rmse_f_sys.append(err_f)
            rmse_f.append(rmse_f_sys)
        return rmse_f

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        rmse_f = self.validate(ip["valid_systems"])
        nf = sum([len(i) for i in rmse_f])
        rmse_f = np.sqrt(sum([sum([j**2 for j in i]) for i in rmse_f]) / nf)
        n_current = 0
        for sys in ip["current_systems"]:
            if sys is not None:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                n_current += len(k)
        ip["learning_curve"].append([n_current, float(rmse_f)])
        if len(ip["candidate_systems"]) == 0:
            return OPIO({
                "remaining_systems": [],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": 0,
                "converged": False,
                "learning_curve": ip["learning_curve"],
            })

        rmse_f = self.validate(ip["candidate_systems"])
        nf = sum([len(i) for i in rmse_f])
        f_max = max([max(i) for i in rmse_f if len(i) > 0])
        f_avg = np.sqrt(sum([sum([j**2 for j in i]) for i in rmse_f]) / nf)
        f_min = min([min(i) for i in rmse_f if len(i) > 0])
        print('max force (eV/A): ', f_max)
        print('avg force (eV/A): ', f_avg)
        print('min force (eV/A): ', f_min)
        if f_max - f_avg <= ip["threshold"] * f_avg:
            return OPIO({
                "remaining_systems": ip["candidate_systems"],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": len(ip["candidate_systems"]),
                "converged": True,
                "learning_curve": ip["learning_curve"],
            })

        mapping = sum([[(i, j) for j in range(len(s))] for i, s in enumerate(rmse_f)], [])
        rmse_f_1d = sum(rmse_f, [])
        sorted_indices = np.argsort(rmse_f_1d)
        indices = [mapping[sorted_indices[i]] for i in range(min(len(sorted_indices), ip["max_selected"]))]

        n = len(ip["candidate_systems"])
        current_systems = [None] * n + ip["current_systems"][n:]
        remaining_systems = [None] * n
        for i in range(n):
            selected = [index[1] for index in indices if index[0] == i]
            if len(selected) > 0:
                path = ip["candidate_systems"][i]
                k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                path0 = ip["current_systems"][i]
                if path0 is None:
                    k0 = k[:0]
                else:
                    k0 = dpdata.LabeledSystem(path0, fmt="deepmd/npy")
                root = str(path)[:str(path).find("iter") + 4]
                target = Path("iter") / path.relative_to(root)
                sum([k[j] for j in selected], k0).to_deepmd_npy_mixed(target)
                current_systems[i] = target
                if len(selected) < len(k):
                    target = path
                    sum([k[j] for j in range(len(k)) if j not in selected], k[:0]).to_deepmd_npy_mixed(target)
                    remaining_systems[i] = target
            else:
                current_systems[i] = ip["current_systems"][i]
                remaining_systems[i] = ip["candidate_systems"][i]

        return OPIO({
                "remaining_systems": remaining_systems,
                "current_systems": current_systems,
                "n_selected": len(indices),
                "n_remaining": len(sorted_indices) - len(indices),
                "converged": False,
                "learning_curve": ip["learning_curve"],
            })
