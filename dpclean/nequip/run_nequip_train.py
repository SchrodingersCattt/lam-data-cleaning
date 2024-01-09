import math
import os
from functools import partial
from pathlib import Path

from dflow.python import OP, OPIO
from dpclean.op import RunTrain

from .utils import deepmd_to_xyz


class RunNequipTrain(RunTrain):
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        import multiprocessing

        import ase
        import yaml
        params = ip["train_params"]
        with multiprocessing.Pool() as pool:
            os.makedirs("train_data", exist_ok=True)
            pool.map(partial(deepmd_to_xyz, outdir="train_data"), ip["train_systems"])

        with open("train.xyz", "w") as fw:
            for sys in ip["train_systems"]:
                with open("train_data/%s.xyz" % sys, "r") as fr:
                    fw.write(fr.read())
        n = len(ase.io.read("train.xyz", index=":"))
        n_val = math.ceil(n*0.05)
        n_train = n - n_val

        params["dataset_file_name"] = "train.xyz"
        params["n_train"] = n_train
        params["n_val"] = n_val

        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "nequip-train input.yaml"
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        return OPIO({
            "model": Path(params["root"]) / params["run_name"],
            "output_files": [Path(params["root"])],
        })
