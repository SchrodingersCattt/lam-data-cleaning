import os
import shutil
from functools import partial
from pathlib import Path

from dpclean.op import Validate

from .utils import deepmd_to_xyz


class NequipValidate(Validate):
    def load_model(self, model: Path):
        self.model = model

    def validate(self, systems, train_params, batch_size):
        import multiprocessing

        import ase
        import dpdata
        import numpy as np
        import yaml

        if os.path.isdir("valid_data"):
            shutil.rmtree("valid_data")
        with multiprocessing.Pool() as pool:
            os.makedirs("valid_data", exist_ok=True)
            pool.map(partial(deepmd_to_xyz, outdir="valid_data"), systems)
        with open("valid.xyz", "w") as fw:
            for sys in systems:
                with open("valid_data/%s.xyz" % sys, "r") as fr:
                    fw.write(fr.read())

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
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            d = dpdata.MultiSystems()
            if mixed_type:
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                d.append(k)

            for k in d:
                rmse_f_sys = []
                rmse_e_sys = []
                natoms_sys = []
                for i in range(len(k)):
                    e_pred = pred_atoms[j].get_potential_energy()
                    f_pred = pred_atoms[j].get_forces()
                    j += 1
                    f_label = k[i].data["forces"][0]
                    e_label = k[i].data["energies"][0]
                    n = f_label.shape[0]
                    rmse_e_sys.append(abs(e_pred - e_label) / n)
                    rmse_f_sys.append(np.sqrt(np.mean((f_pred - f_label)**2)))
                    natoms_sys.append(n)
                rmse_f.append(rmse_f_sys)
                rmse_e.append(rmse_e_sys)
                natoms.append(natoms_sys)
        return rmse_f, rmse_e, None, natoms
