import math
import os
import random
import shutil
from pathlib import Path
from typing import List

import dpdata
from dflow.python import OP, OPIO, Artifact, OPIOSign, Parameter


class SplitDataset(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "dataset": Artifact(Path),
            "ratio_init": Parameter(float, default=0.0),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "systems": Artifact(List[Path]),
            "init_systems": Artifact(List[Path]),
            "n_total": int,
        })

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        init_systems = []

        n_total = 0
        remaining_systems = []
        for f in ip["dataset"].rglob("**/type.raw"):
            path = f.parent
            d = dpdata.MultiSystems()
            mixed_type = len(list(path.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d.load_systems_from_file(path, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                d.append(k)

            selected_systems = dpdata.MultiSystems()
            unselected_systems = dpdata.MultiSystems()
            for s in d:
                n_total += len(s)
                ns = math.floor(len(s)*ip["ratio_init"])
                if random.random() < len(s)*ip["ratio_init"] - ns:
                    ns += 1
                selected_indices = random.sample(range(len(s)), ns)
                unselected_indices = list(set(range(len(s))).difference(selected_indices))
                if len(selected_indices) > 0:
                    selected_systems.append(s.sub_system(selected_indices))
                if len(unselected_indices) > 0:
                    unselected_systems.append(s.sub_system(unselected_indices))

            if len(selected_systems) > 0:
                target = Path("init") / path.relative_to(ip["dataset"])
                if len(selected_systems) == 1:
                    if mixed_type:
                        selected_systems[0].to_deepmd_npy_mixed(target)
                    else:
                        selected_systems[0].to_deepmd_npy(target)
                else:
                    # The multisystem is loaded from one dir, thus we can safely keep one dir
                    selected_systems.to_deepmd_npy_mixed("%s.tmp" % target)
                    fs = os.listdir("%s.tmp" % target)
                    assert len(fs) == 1
                    os.rename(os.path.join("%s.tmp" % target, fs[0]), target)
                    os.rmdir("%s.tmp" % target)
                init_systems.append(target)

            if len(unselected_systems) > 0:
                target = path
                shutil.rmtree(target)
                if len(unselected_systems) == 1:
                    if mixed_type:
                        unselected_systems[0].to_deepmd_npy_mixed(target)
                    else:
                        unselected_systems[0].to_deepmd_npy(target)
                else:
                    # The multisystem is loaded from one dir, thus we can safely keep one dir
                    unselected_systems.to_deepmd_npy_mixed("%s.tmp" % target)
                    fs = os.listdir("%s.tmp" % target)
                    assert len(fs) == 1
                    os.rename(os.path.join("%s.tmp" % target, fs[0]), target)
                    os.rmdir("%s.tmp" % target)
                remaining_systems.append(target)

        return OPIO({
            "systems": remaining_systems,
            "init_systems": init_systems,
            "n_total": n_total,
        })
