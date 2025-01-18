import os
import shutil
from functools import partial
from pathlib import Path

from dpclean.op import Validate

from .utils import deepmd_to_xyz


class NequipValidate(Validate):
    def load_model(self, model: Path):
        self.model = model

    def validate(self, systems, train_params, batch_size, optional_args=None):
        import json
        import multiprocessing

        import ase
        import numpy as np
        import yaml

        with open(self.model / "bias.json", "r") as f:
            energy_bias = json.load(f)
        if os.path.isdir("valid_data"):
            shutil.rmtree("valid_data")
        with multiprocessing.Pool() as pool:
            os.makedirs("valid_data", exist_ok=True)
            pool.map(partial(deepmd_to_xyz, outdir="valid_data", energy_bias=energy_bias), systems)
        with open("valid.xyz", "w") as fw:
            for sys in systems:
                i = 0
                while os.path.isfile("valid_data/%s_%s.xyz" % (sys, i)):
                    with open("valid_data/%s_%s.xyz" % (sys, i), "r") as fr:
                        fw.write(fr.read())
                    i += 1

        params = train_params
        params["dataset_file_name"] = "valid.xyz"
        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "nequip-evaluate --train-dir %s --dataset-config input.yaml --output results.xyz" % self.model 
        if not (isinstance(batch_size, str) and batch_size.startswith("auto")):
            cmd += " --batch-size %s" % batch_size
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        pred_atoms = ase.io.read("results.xyz", index=":")
        rmse_f = []
        rmse_e = []
        natoms = []
        j = 0
        for sys in systems:
            i = 0
            while os.path.isfile("valid_data/%s_%s.xyz" % (sys, i)):
                label_atoms = ase.io.read("valid_data/%s_%s.xyz" % (sys, i), index=":")
                rmse_f_sys = []
                rmse_e_sys = []
                natoms_sys = []
                for k in range(len(label_atoms)):
                    e_pred = pred_atoms[j].get_potential_energy()
                    f_pred = pred_atoms[j].get_forces()
                    j += 1
                    f_label = label_atoms[k].get_forces()
                    e_label = label_atoms[k].get_potential_energy()
                    n = f_label.shape[0]
                    rmse_e_sys.append(abs(e_pred - e_label) / n)
                    rmse_f_sys.append(np.sqrt(np.mean((f_pred - f_label)**2)))
                    natoms_sys.append(n)
                rmse_f.append(rmse_f_sys)
                rmse_e.append(rmse_e_sys)
                natoms.append(natoms_sys)
                i += 1
        return {
            "rmse_epa": rmse_e,
            "rmse_f": rmse_f,
            "natoms": natoms,
        }
