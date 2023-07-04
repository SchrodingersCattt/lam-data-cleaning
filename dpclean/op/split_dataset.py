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
            "init_systems": Artifact(List[Path]),
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
        nframes = []
        for f in ip["dataset"].rglob("**/type.raw"):
            path = f.parent
            k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
            nf = k.get_nframes()
            systems.append(path)
            nframes.append(nf)

        n_total = sum(nframes)
        indices = random.sample(sum([[(i, j) for j in range(n)] for i, n in enumerate(nframes)], []), ip["n_init"])

        init_systems = [None] * len(systems)
        if ip["init_systems"] is not None:
            init_systems += ip["init_systems"]
        remaining_systems = [None] * len(systems)
        for i, path in enumerate(systems):
            selected = [index[1] for index in indices if index[0] == i]
            k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
            if len(selected) > 0:
                target = Path("init") / path.relative_to(ip["dataset"])
                sum([k[j] for j in selected], k[:0]).to_deepmd_npy_mixed(target)
                init_systems[i] = target
            if len(selected) < len(k):
                target = Path("iter") / path.relative_to(ip["dataset"])
                sum([k[j] for j in range(len(k)) if j not in selected], k[:0]).to_deepmd_npy_mixed(target)
                remaining_systems[i] = target

        return OPIO({
            "systems": remaining_systems,
            "init_systems": init_systems,
            "n_total": n_total,
        })
