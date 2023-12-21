import glob
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
        if ip["old_systems"] is not None:
            params["training"]["training_data"]["systems"] = [
                str(s) for s in ip["old_systems"]] + params["training"]["training_data"]["systems"]
            n_old = len(ip["old_systems"])
            n_all = n_old + len(ip["train_params"])
            old_ratio = ip["old_ratio"]
            params["training"]["auto_prob_style"] = "prob_sys_size; 0:%s:%s; %s:%s:%s" % (n_old, old_ratio, n_old, n_all, 1-old_ratio)
        params["training"]["numb_steps"] = int(params["training"]["numb_steps"])
        params["learning_rate"]["decay_steps"] = int(params["learning_rate"]["decay_steps"])
        if params["training"]["numb_steps"] > 1000000:
            params["training"]["numb_steps"] = 1000000
        if params["learning_rate"]["decay_steps"] < 1:
            params["learning_rate"]["decay_steps"] = 1
        if params["learning_rate"]["decay_steps"] > 5000:
            params["learning_rate"]["decay_steps"] = 5000

        with open("input.json", "w") as f:
            json.dump(params, f, indent=2)

        if len(glob.glob("model_[0-9]*.pt")) > 0:  # for restart
            checkpoint = "model_%s.pt" % max([int(f[6:-3]) for f in glob.glob("model_[0-9]*.pt")])
            cmd = 'dp_pt train input.json --restart %s' % checkpoint
        elif ip["model"] is not None:
            cmd = 'dp_pt train input.json --init-model %s' % ip["model"]
        elif ip["finetune_model"] is not None:
            cmd = 'dp_pt train --finetune %s %s input.json' % (ip["finetune_model"], ip["finetune_args"])
        else:
            cmd = 'dp_pt train input.json'
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        return OPIO({
            "model": Path("model.pt"),
            "output_files": [Path("input.json"), Path("lcurve.out")],
        })
