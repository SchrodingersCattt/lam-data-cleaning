import os
from pathlib import Path
from typing import List, Optional, Tuple

import dpdata
import numpy as np
from dpclean.op import SelectSamples, Validate


class DPValidate(Validate):
    def load_model(self, model: Path):
        self.model = model
        from deepmd.infer import DeepPot
        self.dp = DeepPot(model)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coord = coord.reshape([1, -1])
        if cell is not None:
            cell = cell.reshape([1, -1])
        e, f, v = self.dp.eval(coord, cell, atype)
        return e[0], f[0], v[0].reshape([3, 3])

    def validate(self, systems, train_params, batch_size="auto", optional_args=None):
        with open("valid.txt", "w") as f:
            f.write("\n".join([str(sys) for sys in systems]))
        cmd = "dp test -m %s -f valid.txt -n 99999999 -d result" % self.model
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        metrics = {
            "mae_e": [],
            "rmse_e": [],
            "mae_epa": [],
            "rmse_epa": [],
            "mae_f": [],
            "rmse_f": [],
            "mae_v": [],
            "rmse_v": [],
            "mae_vpa": [],
            "rmse_vpa": [],
            "natoms": [],
        }
        e = np.loadtxt("result.e.out") if os.path.exists("result.e.out") else None
        f = np.loadtxt("result.f.out") if os.path.exists("result.f.out") else None
        v = np.loadtxt("result.v.out") if os.path.exists("result.v.out") else None
        i_e = 0
        i_f = 0
        for sys in systems:
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            d = dpdata.MultiSystems()
            if mixed_type:
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                d.append(k)

            for k in d:
                metrics_sys = {key: [] for key in metrics}
                for i in range(len(k)):
                    n = k.get_natoms()

                    if "energies" in k.data:
                        metrics_sys["mae_e"].append(np.mean(np.abs(e[i_e][0] - e[i_e][1])))
                        metrics_sys["rmse_e"].append(np.sqrt(np.mean((e[i_e][0] - e[i_e][1])**2)))
                        metrics_sys["mae_epa"].append(np.mean(np.abs(e[i_e][0] - e[i_e][1])) / n)
                        metrics_sys["rmse_epa"].append(np.sqrt(np.mean((e[i_e][0] - e[i_e][1])**2)) / n)
                    if "forces" in k.data:
                        metrics_sys["mae_f"].append(np.mean(np.abs(f[i_f:i_f+n, 0:3] - f[i_f:i_f+n, 3:6])))
                        metrics_sys["rmse_f"].append(np.sqrt(np.mean((f[i_f:i_f+n, 0:3] - f[i_f:i_f+n, 3:6])**2)))
                    if "virials" in k.data:
                        metrics_sys["mae_v"].append(np.mean(np.abs(v[i_e, 0:9] - v[i_e, 9:18])))
                        metrics_sys["rmse_v"].append(np.sqrt(np.mean((v[i_e, 0:9] - v[i_e, 9:18])**2)))
                        metrics_sys["mae_vpa"].append(np.mean(np.abs(v[i_e, 0:9] - v[i_e, 9:18])) / n)
                        metrics_sys["rmse_vpa"].append(np.sqrt(np.mean((v[i_e, 0:9] - v[i_e, 9:18])**2)) / n)
                    metrics_sys["natoms"].append(n)
                    i_e += 1
                    i_f += n
                print("System: %s metrics: %s" % (sys, metrics_sys))
                for key in metrics:
                    if len(metrics_sys[key]) > 0:
                        metrics[key].append(metrics_sys[key])
        return {key: np.array(value) for key, value in metrics.items() if len(value) > 0}


class DPSelectSamples(SelectSamples, DPValidate):
    pass
