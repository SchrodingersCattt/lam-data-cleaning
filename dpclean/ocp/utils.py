import os
from pathlib import Path


def deepmd_to_ase(sys: Path, outdir: str, max_frames=None):
    import ase
    import dpdata

    mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
    if mixed_type:
        d = dpdata.MultiSystems()
        d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
        k = d[0]
    else:
        k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
    k.data["cells"] = k.data["cells"].astype(float)
    k.data["coords"] = k.data["coords"].astype(float)
    k.data["energies"] = k.data["energies"].astype(float)
    k.data["forces"] = k.data["forces"].astype(float)
    if max_frames is not None:
        k.shuffle()
        k = k[:max_frames]
    a = k.to_ase_structure()
    target = "%s/%s.json" % (outdir, sys)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    ase.io.write(target, a, format="json")
