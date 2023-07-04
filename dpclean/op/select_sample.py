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
                "selected_systems": Artifact(List[Path]),
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
            k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
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
                rmse_f.append(err_f)
        return rmse_f

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        self.load_model(ip["model"])
        rmse_f = self.validate(ip["valid_systems"])
        rmse_f = np.sqrt(np.mean(np.square(rmse_f)))
        ip["learning_curve"].append([len(ip["current_systems"]), float(rmse_f)])
        if len(ip["candidate_systems"]) == 0:
            return OPIO({
                "selected_systems": [],
                "remaining_systems": [],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": 0,
                "converged": False,
                "learning_curve": ip["learning_curve"],
            })

        rmse_f = self.validate(ip["candidate_systems"])
        f_max = max(rmse_f)
        f_ave = np.sqrt(np.mean(np.square(rmse_f)))
        f_min = min(rmse_f)
        print('max force (eV/A): ', f_max)
        print('ave force (eV/A): ', f_ave)
        print('min force (eV/A): ', f_min)
        if f_max - f_ave <= ip["threshold"] * f_ave:
            return OPIO({
                "selected_systems": [],
                "remaining_systems": ip["candidate_systems"],
                "current_systems": ip["current_systems"],
                "n_selected": 0,
                "n_remaining": len(ip["candidate_systems"]),
                "converged": True,
                "learning_curve": ip["learning_curve"],
            })

        sorted_indices = np.argsort(rmse_f)
        selected_systems = []
        remaining_systems = []
        for i in range(len(sorted_indices)):
            if i < ip["max_selected"]:
                selected_systems.append(ip["candidate_systems"][sorted_indices[i]])
            else:
                remaining_systems.append(ip["candidate_systems"][sorted_indices[i]])

        return OPIO({
                "selected_systems": selected_systems,
                "remaining_systems": remaining_systems,
                "current_systems": ip["current_systems"] + selected_systems,
                "n_selected": len(selected_systems),
                "n_remaining": len(remaining_systems),
                "converged": False,
                "learning_curve": ip["learning_curve"],
            })
