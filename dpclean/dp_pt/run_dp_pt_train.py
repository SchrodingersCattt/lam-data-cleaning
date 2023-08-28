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

        train_dir = Path("train")
        train_dir.mkdir(exist_ok=True)
        os.chdir("train")
        with open("input.json", "w") as f:
            json.dump(params, f, indent=2)

        if os.path.exists("model.pt"):  # for restart
            cmd = 'dp_pt train input.json --restart model.pt'
        elif ip["model"] is not None:
            cmd = 'dp_pt train input.json --init-model %s' % ip["model"]
        elif ip["finetune_model"] is not None:
            cmd = 'dp_pt train --finetune %s %s input.json' % (ip["finetune_model"], ip["finetune_args"])
        else:
            cmd = 'dp_pt train input.json'
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd
        os.chdir("..")

        return OPIO({
            "model": train_dir / "model.pt",
            "output_dir": train_dir,
        })
