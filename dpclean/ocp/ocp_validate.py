import json
import os
import shutil
from functools import partial
from pathlib import Path

from dpclean.op import Validate

from .utils import deepmd_to_ase


class OCPValidate(Validate):
    def load_model(self, model: Path):
        self.model = model

    def validate(self, systems, train_params, batch_size):
        import multiprocessing

        import ase
        import numpy as np
        import yaml

        checkpoint = self.model
        params = train_params
        if "scale_dict" in params:
            with open("scale.json", "w") as f:
                json.dump(params.pop("scale_dict"), f, indent=4)
            params["model"]["scale_file"] = "scale.json"

        if os.path.isdir("valid_data"):
            shutil.rmtree("valid_data")
        with multiprocessing.Pool() as pool:
            os.makedirs("valid_data", exist_ok=True)
            pool.map(partial(deepmd_to_ase, outdir="valid_data"), systems)

        valid_dataset = params["dataset"][0] if len(params["dataset"]) > 0 else {}
        valid_dataset.update({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"] = []
        params["dataset"].append(valid_dataset)
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["dataset"].append({"src": os.path.abspath("valid_data"), "pattern": "**/*.json", "a2g_args": {"r_energy": True, "r_forces": True}})
        params["task"]["dataset"] = "ase_read_multi"
        if not (isinstance(batch_size, str) and batch_size.startswith("auto")):
            params["optim"]["eval_batch_size"] = int(batch_size)
        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))

        cmd = "python -m main --mode predict --config-yml input.yaml --checkpoint %s" % checkpoint
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        result = np.load(Path("results") / max(os.listdir("results")) / "s2ef_predictions.npz", allow_pickle=True)
        energies = []
        forces = []
        indices = []
        for f in Path("valid_data").rglob("*.json"):
            sys = "_".join(str(f.relative_to("valid_data")).split("_")[:-2])
            i = int(str(f).split("_")[-2])
            j = int(str(f).split("_")[-1][:-5])
            for k, atoms in enumerate(ase.io.read(f, index=":")):
                energies.append(atoms.get_potential_energy())
                forces.append(atoms.get_forces())
                indices.append((sys, i, 30*j+k))

        results = {}
        chunk_idx = [0] + list(result["chunk_idx"]) + [result["forces"].shape[0]]
        for i, id in enumerate(result["ids"]):
            j = int(id.split("_")[0])
            e_pred = result["energy"][i]
            e_label = energies[j]
            f_pred = result["forces"][chunk_idx[i]:chunk_idx[i+1],:]
            f_label = forces[j]
            natoms = f_label.shape[0]
            results[indices[j][0]] = results.get(indices[j][0], {})
            results[indices[j][0]][indices[j][1]] = results[indices[j][0]].get(indices[j][1], {})
            results[indices[j][0]][indices[j][1]][indices[j][2]] = {
                "rmse_e": abs(e_pred - e_label) / natoms,
                "rmse_f": np.sqrt(np.mean((f_pred - f_label)**2)),
                "natoms": natoms,
            }

        rmse_e = []
        rmse_f = []
        natoms = []
        for sys in systems:
            sys = str(sys)
            if sys.startswith("/"):
                sys = sys[1:]
            for i in sorted(results[sys]):
                rmse_f_sys = []
                rmse_e_sys = []
                natoms_sys = []
                for j in sorted(results[sys][i]):
                    rmse_e_sys.append(results[sys][i][j]["rmse_e"])
                    rmse_f_sys.append(results[sys][i][j]["rmse_f"])
                    natoms_sys.append(results[sys][i][j]["natoms"])
                rmse_e.append(rmse_e_sys)
                rmse_f.append(rmse_f_sys)
                natoms.append(natoms_sys)
        return rmse_f, rmse_e, None, natoms
