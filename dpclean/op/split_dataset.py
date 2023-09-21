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
            "init_systems": Artifact(List[Path]),
            "n_init": Parameter(int, default=0),
            "select_type": Parameter(str, default="global"),
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
        systems = []
        nframes = []
        for f in ip["dataset"].rglob("**/type.raw"):
            path = f.parent
            d = dpdata.MultiSystems()
            mixed_type = len(list(path.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d.load_systems_from_file(path, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                d.append(k)
            nf = d.get_nframes()
            systems.append(path)
            nframes.append(nf)

        n_total = sum(nframes)
        if ip["select_type"] == "global":
            indices = random.sample(sum([[(i, j) for j in range(n)] for i, n in enumerate(nframes)], []), ip["n_init"])
        elif ip["select_type"] == "system":
            indices = []
            for i in range(len(systems)):
                n = math.ceil(nframes[i]*ip["ratio_init"])
                indices.extend([(i, j) for j in random.sample(range(nframes[i]), n)])

        init_systems = []
        if ip["init_systems"] is not None:
            init_systems += ip["init_systems"]
        remaining_systems = []
        for i, path in enumerate(systems):
            selected = [index[1] for index in indices if index[0] == i]
            d = dpdata.MultiSystems()
            mixed_type = len(list(path.glob("*/real_atom_types.npy"))) > 0
            if mixed_type:
                d.load_systems_from_file(path, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(path, fmt="deepmd/npy")
                d.append(k)

            selected_systems = dpdata.MultiSystems()
            unselected_systems = dpdata.MultiSystems()
            cnt = 0
            for k in d:
                selected_indices = [j for j in range(len(k)) if cnt+j in selected]
                unselected_indices = [j for j in range(len(k)) if cnt+j not in selected]
                if len(selected_indices) > 0:
                    selected_systems.append(k.sub_system(selected_indices))
                if len(unselected_indices) > 0:
                    unselected_systems.append(k.sub_system(unselected_indices))
                cnt += len(k)

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
