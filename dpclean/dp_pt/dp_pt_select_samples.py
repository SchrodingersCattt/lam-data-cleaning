import os
from pathlib import Path
from typing import List, Optional, Tuple

import dpdata
import numpy as np
from dpclean.op import SelectSamples, Validate


class DPPTValidate(Validate):
    def load_model(self, model: Path):
        self.model = model
        from deepmd_pt.infer.deep_eval import DeepPot
        self.dp = DeepPot(model)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coord = coord.reshape([1, -1, 3])
        if cell is not None:
            cell = cell.reshape([1, 3, 3])
        atype = atype.reshape([1, -1])
        e, f, v = self.dp.eval(coord, cell, atype, infer_batch_size=1)
        return e.reshape([1])[0], f.reshape([-1, 3]), v.reshape([3, 3])

    def validate(self, systems, train_params, batch_size="auto", optional_args=None):
        with open("valid.txt", "w") as f:
            f.write("\n".join([str(sys) for sys in systems]))
        command = (optional_args or {}).get("command", "dp_pt")
        cmd = "%s test -m %s -f valid.txt -n 99999999 -d result -b %s" % (command, self.model, batch_size)
        print("Run command '%s'" % cmd)
        ret = os.system(cmd)
        assert ret == 0, "Command '%s' failed" % cmd

        rmse_f = []
        rmse_e = []
        rmse_v = []
        natoms = []
        e = np.loadtxt("result.e.out")
        f = np.loadtxt("result.f.out")
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
                rmse_f_sys = []
                rmse_e_sys = []
                rmse_v_sys = []
                natoms_sys = []
                for i in range(len(k)):
                    force0 = k[i].data["forces"][0]
                    energy0 = k[i].data["energies"][0]
                    virial0 = k[i].data["virials"][0] if "virials" in k[i].data else None
                    n = force0.shape[0]

                    err_e = abs(e[i_e][0] - e[i_e][1]) / n
                    err_f = np.sqrt(np.average((f[i_f:i_f+n, 0:3] - f[i_f:i_f+n, 3:6])**2))
                    err_v = None
                    if virial0 is not None:
                        err_v = np.sqrt(np.average((v[i_e, 0:9].reshape([3, 3]) - v[i_e, 9:18].reshape([3, 3]))**2)) / n
                    i_e += 1
                    i_f += n
                    print("System: %s frame: %s rmse_e: %s rmse_f: %s rmse_v: %s" % (sys, i, err_e, err_f, err_v))
                    rmse_f_sys.append(err_f)
                    rmse_e_sys.append(err_e)
                    if err_v is not None:
                        rmse_v_sys.append(err_v)
                    natoms_sys.append(n)
                rmse_f.append(rmse_f_sys)
                rmse_e.append(rmse_e_sys)
                if len(rmse_v_sys) > 0:
                    rmse_v.append(rmse_v_sys)
                natoms.append(natoms_sys)
        return rmse_f, rmse_e, rmse_v if len(rmse_v) > 0 else None, natoms


class DPPTSelectSamples(SelectSamples, DPPTValidate):
    pass
