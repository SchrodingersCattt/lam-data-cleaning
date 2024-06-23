from pathlib import Path
from typing import List

import dpdata
from dflow.python import OP, OPIO, Artifact, OPIOSign, NestedDict


class Statistics(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "dataset": Artifact(NestedDict[Path]),
        }) 

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "n_subsets": int,
            "size_list": List[int],
        })

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        size_list = []
        for subset in ip["dataset"]:
            size = 0
            for sys in subset:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                size += len(k)
            size_list.append(size)

        return OPIO({
            "n_subsets": len(ip["dataset"]),
            "size_list": size_list,
        })
