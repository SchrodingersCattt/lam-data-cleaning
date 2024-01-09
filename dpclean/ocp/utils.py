import math
import os
from pathlib import Path


def deepmd_to_ase(sys: Path, outdir: str, max_frames=None):
    import ase
    import dpdata

    mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
    if mixed_type:
        d = dpdata.MultiSystems()
        d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
    else:
        d = dpdata.MultiSystems()
        k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
        d.append(k)

    for i, k in enumerate(d):
        # Gemnet only supports natoms > 2
        if k.get_natoms() <= 2:
            continue
        k.data["cells"] = k.data["cells"].astype(float)
        k.data["coords"] = k.data["coords"].astype(float)
        k.data["energies"] = k.data["energies"].astype(float)
        k.data["forces"] = k.data["forces"].astype(float)
        if max_frames is not None:
            k.shuffle()
            k = k[:max_frames]
        a = k.to_ase_structure()

        for atoms in a:
            if atoms.pbc is False or atoms.pbc is 0 or (hasattr(atoms.pbc, "__iter__") and not any(atoms.pbc)):
                atoms.set_cell([10000.0, 10000.0, 10000.0])
                atoms._calc.atoms = atoms

        for j in range(math.ceil(len(a)/30)):
            target = "%s/%s_%s_%s.json" % (outdir, sys, i, j)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            ase.io.write(target, a[30*j:30*j+30], format="json")
