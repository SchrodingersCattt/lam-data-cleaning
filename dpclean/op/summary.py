import os
from pathlib import Path
from typing import List

from dflow.python import OP, OPIO, Artifact, OPIOSign, Parameter


class Summary(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "size_list": Parameter(List[int], default=[]),
            "results": Parameter(List[dict], default=[]),
            "zero_result": Parameter(dict, default=None),
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
        if ip["zero_result"] is not None:
            ip["size_list"].insert(0, 1)
            for k, v in ip["zero_result"].items():
                lcurve[k] = lcurve.get(k, [])
                lcurve[k].append(v)
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

        lcurve["nsamples"] = ip["size_list"]
        return OPIO({
            "lcurve": lcurve,
            "lcurve_image": Path("lcurve.png"),
        })
