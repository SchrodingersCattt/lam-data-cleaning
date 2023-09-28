import json
import os

from dflow.python import OP, OPIO
from dpclean.op import RunTrain
from pathlib import Path


class RunDPTrain(RunTrain):
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
            cmd = 'dp train --init-frz-model %s input.json && dp freeze -o graph.pb' % ip["model"]
        elif ip["finetune_model"] is not None:
            cmd = 'dp train --finetune %s %s input.json && dp freeze -o graph.pb' % (ip['finetune_model'], ip["finetune_args"])
        else:
            cmd = 'dp train input.json && dp freeze -o graph.pb'
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        return OPIO({
            "model": Path("graph.pb"),
            "output_files": [Path("input.json"), Path("lcurve.out")],
        })
