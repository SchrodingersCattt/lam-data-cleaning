import math
import os
from functools import partial
from pathlib import Path
from typing import List

from dflow.python import OP, OPIO, Artifact, OPIOSign
from dpclean.op import RunTrain

from .utils import deepmd_to_ase


class RunOCPTrain(RunTrain):
    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "model": Artifact(List[Path]),
                "output_files": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        import multiprocessing

        import yaml
        params = ip["train_params"]
        with multiprocessing.Pool() as pool:
            os.makedirs("train_data", exist_ok=True)
            pool.map(partial(deepmd_to_ase, outdir="train_data"), ip["train_systems"])
            os.makedirs("valid_data", exist_ok=True)
            pool.map(partial(deepmd_to_ase, outdir="valid_data", max_frames=math.ceil(1000/len(ip["valid_systems"]))), ip["valid_systems"])

        if len(params["dataset"]) > 0:
            train_dataset = params["dataset"][0]
        else:
            train_dataset = {}
        train_dataset.update({"src": os.path.abspath("train_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"] = []
        params["dataset"].append(train_dataset)
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["task"]["dataset"] = "ase_read_multi"
        params["optim"]["max_epochs"] = int(params["optim"]["max_epochs"])
        params["optim"]["eval_every"] = int(params["optim"]["eval_every"])

        extra_files = []
        if ip["optional_artifact"] is not None and "scale_file" in ip["optional_artifact"]:
            scale_file = ip["optional_artifact"]["scale_file"]
            scale_file.rename(scale_file.name)
            params["model"]["scale_file"] = scale_file.name
            extra_files.append(Path(scale_file.name))
        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "python -m main --mode train --config-yml input.yaml"
        if ip["finetune_model"] is not None:
            cmd += " --checkpoint %s" % ip["finetune_model"]
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd
        checkpoint = Path("checkpoints") / max(os.listdir("checkpoints")) / "checkpoint.pt"

        return OPIO({
            "model": [checkpoint, Path("input.yaml")] + extra_files,
            "output_files": [Path("checkpoints"), Path("logs"), Path("results")],
        })
