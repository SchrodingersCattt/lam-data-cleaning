import os
from pathlib import Path

import numpy as np


def deepmd_to_xyz(sys: Path, outdir: str, max_frames=None, energy_bias=None):
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
        # allegro will complain no neighbor
        if k.nopbc and k.get_natoms() <= 1:
            continue
        k.data["cells"] = k.data["cells"].astype(float)
        k.data["coords"] = k.data["coords"].astype(float)
        k.data["energies"] = k.data["energies"].astype(float)
        if energy_bias is not None:
            k.data["energies"] -= sum([energy_bias.get(a, 0.0)*n for a, n in zip(k["atom_names"], k["atom_numbs"]) if n > 0])
        k.data["forces"] = k.data["forces"].astype(float)
        k.data.pop("virials", None)
        if max_frames is not None:
            k.shuffle()
            k = k[:max_frames]
        a = k.to_ase_structure()

        for atoms in a:
            if atoms.pbc is False or atoms.pbc is 0 or (hasattr(atoms.pbc, "__iter__") and not any(atoms.pbc)):
                atoms.set_pbc(True)
                pos = atoms.get_positions()
                max_len = np.sqrt((max(pos[:,0])-min(pos[:,0]))**2+(max(pos[:,1])-min(pos[:,1]))**2+(max(pos[:,2])-min(pos[:,2]))**2)
                box = max_len + 7
                atoms.set_cell([box, box, box])
                atoms._calc.atoms = atoms

        target = "%s/%s_%s.xyz" % (outdir, sys, i)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        ase.io.write(target, a, format='extxyz')
