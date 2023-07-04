import json
import os

from dflow.python import OP, OPIO
from dpclean.op import RunTrain
from pathlib import Path


class RunDPPTTrain(RunTrain):
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        params = ip["train_params"]
        params["training"]["training_data"]["systems"] = [
            str(s) for s in ip["train_systems"]]
        params["training"]["validation_data"]["systems"] = [
            str(s) for s in ip["valid_systems"]]

        with open("input.json", "w") as f:
            json.dump(params, f, indent=2)

        if ip["model"] is not None:
            cmd = 'dp_pt train input.json %s' % ip["model"]
        elif ip["finetune_model"] is not None:
            cmd = 'dp_pt train --finetune %s input.json' % ip["finetune_model"]
        else:
            cmd = 'dp_pt train input.json'
        ret = os.system(cmd)
        assert ret == 0, "Command %s failed" % cmd

        return OPIO({
            "model": Path("model.pt"),
            "output_dir": Path("."),
        })
