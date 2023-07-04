import os
import random
from pathlib import Path
from typing import List

import dpdata
from dflow.python import OP, OPIO, Artifact, OPIOSign, Parameter


class SplitDataset(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "dataset": Artifact(Path),
            "n_init": Parameter(int, default=0),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "systems": Artifact(List[Path]),
            "init_systems": Artifact(List[Path]),
            "n_total": int,
        })

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        systems = []
        for f in ip["dataset"].rglob("**/type.raw"):
            path = f.parent
            k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
            nf = k.get_nframes()

            for i in range(nf):
                target = path.relative_to(ip["dataset"]) / str(i)
                print("Save system to ", target)
                k[i].to_deepmd_npy_mixed(target)
                systems.append(target)

        init_systems = random.sample(systems, ip["n_init"])
        return OPIO({
            "systems": [s for s in systems if s not in init_systems],
            "init_systems": init_systems,
            "n_total": len(systems),
        })
