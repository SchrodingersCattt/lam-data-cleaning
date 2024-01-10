import json
import math
import os
from functools import partial
from pathlib import Path

from dflow.python import OP, OPIO
from dpclean.op import RunTrain

from .utils import deepmd_to_ase


class RunOCPTrain(RunTrain):
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        import multiprocessing
        import shutil

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
            pool.map(partial(deepmd_to_ase, outdir="train_data", energy_bias=energy_bias), ip["train_systems"])
            os.makedirs("valid_data", exist_ok=True)
            pool.map(partial(deepmd_to_ase, outdir="valid_data", max_frames=math.ceil(1000/len(ip["valid_systems"])), energy_bias=energy_bias), ip["valid_systems"])

        energies = []
        for f in Path("train_data").rglob("*.json"):
            for atoms in ase.io.read(f, index=":"):
                energies.append(atoms.get_potential_energy())

        params = ip["train_params"]
        train_dataset = {
            "src": os.path.abspath("train_data"),
            "pattern": "**/*.json",
            "a2g_args": {"r_energy": True, "r_forces": True},
            "normalize_labels": True,
            "target_mean": float(np.mean(energies)),
            "target_std": float(np.std(energies)),
            "grad_target_mean": 0.0,
            "grad_target_std": 1.0
        }
        params["dataset"] = []
        params["dataset"].append(train_dataset)
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["task"]["dataset"] = "ase_read_multi"
        params["optim"]["max_epochs"] = int(params["optim"]["max_epochs"])
        params["optim"]["eval_every"] = int(params["optim"]["eval_every"])

        if "scale_dict" in params:
            with open("scale.json", "w") as f:
                json.dump(params.pop("scale_dict"), f, indent=4)
            params["model"]["scale_file"] = "scale.json"

        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "python -m main --mode train --config-yml input.yaml"
        if ip["pretrained_model"] is not None:
            cmd += " --checkpoint %s" % ip["pretrained_model"]
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd
        checkpoint = max(Path("checkpoints").glob("*/checkpoint.pt"))
        checkpoint_dir = checkpoint.parent
        with open(checkpoint_dir / "bias.json", "w") as f:
            json.dump(energy_bias, f, indent=4)
        shutil.copy("input.yaml", checkpoint_dir / "input.yaml")
        if os.path.isfile("scale.json"):
            shutil.copy("scale.json", checkpoint_dir / "scale.json")

        return OPIO({
            "model": checkpoint_dir,
            "output_files": [Path("checkpoints"), Path("logs"), Path("results")],
        })
