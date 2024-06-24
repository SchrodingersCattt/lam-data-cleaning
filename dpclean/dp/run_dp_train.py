import json
import math
import os

import dpdata
from dflow.python import OP, OPIO
from dpclean.op import RunTrain
from pathlib import Path


class RunDPTrain(RunTrain):
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        BACKEND = ip["optional_args"]["backend"]
        print(BACKEND)
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
            params["learning_rate"]["decay_steps"] = eval(params["learning_rate"]["decay_steps"], {"n": nf, "nsteps": params["training"]["numb_steps"], "math": math})
        if isinstance(params["learning_rate"]["start_lr"], str):
            params["learning_rate"]["start_lr"] = eval(params["learning_rate"]["start_lr"], {"n": nf, "nsteps": params["training"]["numb_steps"], "math": math})

        with open("input.json", "w") as f:
            json.dump(params, f, indent=2)

        if BACKEND == "tf":
            if os.path.exists("checkpoint"):  # for restart
                cmd = 'dp train --restart model.ckpt input.json'
            elif ip["model"] is not None:
                cmd = 'dp train --init-frz-model %s input.json && dp freeze -o graph.pb' % ip["model"]
            elif ip["pretrained_model"] is not None:
                cmd = 'dp train --finetune %s %s input.json && dp freeze -o graph.pb' % (ip['pretrained_model'], ip["finetune_args"])
            else:
                cmd = 'dp train input.json && dp freeze -o graph.pb'

            print("Run command '%s'" % cmd)
            ret = os.system(cmd)
            assert ret == 0, "Command '%s' failed" % cmd

            return OPIO({
                "model": Path("graph.pb"),
                "output_files": [Path("input.json"), Path("lcurve.out")],
            })

        elif BACKEND == "pt" or "null":
            if os.path.exists("checkpoint"):  # for restart
                cmd = 'dp --pt train --restart model.ckpt.pt input.json'
            elif ip["pretrained_model"] is not None:
                cmd = 'dp --pt train input.json --finetune %s %s' % (ip['pretrained_model'], ip["finetune_args"])
            else:
                cmd = 'dp --pt train input.json'

            print("Run command '%s'" % cmd)
            ret = os.system(cmd)
            assert ret == 0, "Command '%s' failed" % cmd

            return OPIO({
                "model": Path("model.ckpt.pt"),
                "output_files": [Path("input.json"), Path("lcurve.out")],
            })

        else:
            raise ValueError(f"Unsupported backend: {BACKEND}. Expected 'tf', 'pt', or None for pt by default.")
