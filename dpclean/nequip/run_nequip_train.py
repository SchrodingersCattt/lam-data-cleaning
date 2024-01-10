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
        import json
        import multiprocessing

        import ase
        import dpdata
        import numpy as np
        import yaml
        formulas = []
        energies = []
        for system in ip["train_systems"]:
            mixed_type = len(list(system.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d = dpdata.MultiSystems()
                d.load_systems_from_file(system, fmt="deepmd/npy/mixed")
            else:
                d = dpdata.MultiSystems()
                k = dpdata.LabeledSystem(system, fmt="deepmd/npy")
                d.append(k)
            for k in d:
                formulas.append({k: v for k, v in zip(k["atom_names"], k["atom_numbs"]) if v > 0})
                energies.append(np.mean(k["energies"]))
        elements = set()
        for f in formulas:
            elements.update(list(f))
        natoms = []
        for f in formulas:
            natoms.append([f.get(e, 0) for e in elements])
        coef, _, _, _ = np.linalg.lstsq(natoms, energies, rcond=None)
        energy_bias = {k: v for k, v in zip(elements, coef)}
        print("energy_bias: ", energy_bias)

        with multiprocessing.Pool() as pool:
            os.makedirs("train_data", exist_ok=True)
            pool.map(partial(deepmd_to_xyz, outdir="train_data", energy_bias=energy_bias), ip["train_systems"])

        with open("train.xyz", "w") as fw:
            for sys in ip["train_systems"]:
                i = 0
                while os.path.isfile("train_data/%s_%s.xyz" % (sys, i)):
                    with open("train_data/%s_%s.xyz" % (sys, i), "r") as fr:
                        fw.write(fr.read())
                    i += 1
        n = len(ase.io.read("train.xyz", index=":"))
        n_val = math.ceil(n*0.05)
        n_train = n - n_val

        params = ip["train_params"]
        params["dataset_file_name"] = "train.xyz"
        params["n_train"] = n_train
        params["n_val"] = n_val

        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "nequip-train input.yaml"
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd
        train_dir = Path(params["root"]) / params["run_name"]
        with open(train_dir / "bias.json", "w") as f:
            json.dump(energy_bias, f, indent=4)

        return OPIO({
            "model": train_dir,
            "output_files": [Path(params["root"])],
        })
