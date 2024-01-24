import glob
import json
import math
import os

import dpdata
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
        nf = 0
        for sys in ip["train_systems"]:
            k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            nf += len(k)
        if isinstance(params["training"]["numb_steps"], str):
            params["training"]["numb_steps"] = eval(params["training"]["numb_steps"], {"n": nf, "math": math})
        if isinstance(params["learning_rate"]["decay_steps"], str):
            params["learning_rate"]["decay_steps"] = eval(params["learning_rate"]["decay_steps"], {"n": nf, "math": math})

        with open("input.json", "w") as f:
            json.dump(params, f, indent=2)

        command = ip["optional_args"].get("command", "dp_pt")
        if len(glob.glob("model_[0-9]*.pt")) > 0:  # for restart
            checkpoint = "model_%s.pt" % max([int(f[6:-3]) for f in glob.glob("model_[0-9]*.pt")])
            cmd = '%s train input.json --restart %s' % (command, checkpoint)
        elif ip["model"] is not None:
            cmd = '%s train input.json --init-model %s' % (command, ip["model"])
        elif ip["pretrained_model"] is not None:
            cmd = '%s train --finetune %s %s input.json' % (command, ip["pretrained_model"], ip["finetune_args"])
        else:
            cmd = '%s train input.json' % command
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        return OPIO({
            "model": Path("model.pt"),
            "output_files": [Path("input.json"), Path("lcurve.out")],
        })
