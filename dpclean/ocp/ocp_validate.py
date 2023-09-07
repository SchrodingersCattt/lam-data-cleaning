import os
from functools import partial
from pathlib import Path
from typing import List

from dflow.python import OP, OPIO, Artifact, OPIOSign

from .utils import deepmd_to_ase


class OCPValidate(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "valid_systems": Artifact(List[Path]),
                "model": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "results": dict,
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        import multiprocessing

        import dpdata
        import numpy as np
        import yaml

        checkpoint = ip["model"][0]
        with open(ip["model"][1], "r") as f:
            params = yaml.full_load(f.read())
        for f in ip["model"][2:]:
            Path(f.name).symlink_to(f)

        with multiprocessing.Pool() as pool:
            os.makedirs("valid_data", exist_ok=True)
            pool.map(partial(deepmd_to_ase, outdir="valid_data"), ip["valid_systems"])

        params["dataset"] = []
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["task"]["dataset"] = "ase_read_multi"
        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "python -m main --mode predict --config-yml input.yaml --checkpoint %s" % checkpoint
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        result = np.load(Path("results") / max(os.listdir("results")) / "s2ef_predictions.npz", allow_pickle=True)
        energies = []
        forces = []
        for f in Path("valid_data").rglob("*.json"):
            sys = Path(str(f)[10:-5])
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d = dpdata.MultiSystems()
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
                k = d[0]
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
            for i in range(len(k)):
                energies.append(k["energies"][i])
                forces.append(k["forces"][i])
        sum_err_e = 0.
        sum_err_f = 0.
        sum_natoms = 0
        chunk_idx = [0] + list(result["chunk_idx"]) + [result["forces"].shape[0]]
        for i, id in enumerate(result["ids"]):
            j = int(id.split("_")[0])
            e_pred = result["energy"][i]
            e_label = energies[j]
            f_pred = result["forces"][chunk_idx[i]:chunk_idx[i+1],:]
            f_label = forces[j]
            natoms = f_label.shape[0]
            sum_err_e += ((e_pred - e_label) / natoms)**2 * natoms
            sum_err_f += np.mean((f_pred - f_label)**2) * natoms
            sum_natoms += natoms
        rmse_e = np.sqrt(sum_err_e / sum_natoms)
        rmse_f = np.sqrt(sum_err_f / sum_natoms)
        return OPIO({
            "results": {"rmse_f": float(rmse_f), "rmse_e": float(rmse_e)},
        })
