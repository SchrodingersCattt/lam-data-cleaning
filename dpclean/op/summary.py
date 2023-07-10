import os
from pathlib import Path
from typing import List

from dflow.python import OP, OPIO, Artifact, OPIOSign


class Summary(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "size_list": List[int],
            "results": List[dict],
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "lcurve": dict,
            "lcurve_image": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        lcurve = {}
        for res in ip["results"]:
            for k, v in res.items():
                lcurve[k] = lcurve.get(k, [])
                lcurve[k].append(v)

        os.system("pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
        from matplotlib import pyplot
        pyplot.figure(figsize=(10,10))
        for i, k in enumerate(lcurve):
            pyplot.subplot(len(lcurve), 1, i + 1)
            pyplot.xscale("log")
            pyplot.yscale("log")
            pyplot.plot(ip["size_list"], lcurve[k])
            pyplot.xlabel('Number of samples')
            pyplot.ylabel(k)
        pyplot.legend()
        pyplot.savefig("lcurve.png")

        lcurve["n_samples"] = ip["size_list"]
        return OPIO({
            "lcurve": lcurve,
            "lcurve_image": Path("lcurve.png"),
        })
