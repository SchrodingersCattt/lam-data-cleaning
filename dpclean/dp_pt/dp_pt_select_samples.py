from pathlib import Path
from typing import List, Tuple

import numpy as np
from dpclean.op import SelectSamples


def infer_model(
    model,
    coord,
    cell,
    atype,
    type_split: bool=True,
):
    import torch
    from deepmd_pt.utils.env import DEVICE
    from deepmd_pt.utils.preprocess import Region3D, make_env_mat

    rcut = model.descriptor.rcut
    sec = model.descriptor.sec
    if cell is not None:
        region = Region3D(cell)
    else:
        region = None
    # inputs: coord, atype, regin; rcut, sec
    selected, selected_loc, selected_type, merged_coord_shift, merged_mapping = \
        make_env_mat(coord, atype, region, rcut, sec, type_split=type_split)
    # add batch dim
    [batch_coord, batch_atype, batch_shift, batch_mapping, batch_selected, batch_selected_loc, batch_selected_type] = \
        [torch.unsqueeze(ii, 0).to(DEVICE) if not isinstance(ii, list) else [torch.unsqueeze(kk, 0).to(DEVICE) for kk in ii] for ii in \
        [coord, atype, merged_coord_shift, merged_mapping, selected, selected_loc, selected_type]]
    # inference, assumes pbc
    ret = model(
        batch_coord, batch_atype, None,
        batch_mapping, batch_shift,
        batch_selected, batch_selected_type, batch_selected_loc,
        box=cell,
    )
    # remove the frame axis
    ret1 = {}
    for kk, vv in ret.items():
        ret1[kk] = vv[0]
    return ret1


class DPPTSelectSamples(SelectSamples):
    def load_model(self, model: Path):
        import torch
        from deepmd_pt.model.model import get_model
        from deepmd_pt.train.wrapper import ModelWrapper
        from deepmd_pt.utils.env import DEVICE, JIT

        state_dict = torch.load(model)
        model_params = state_dict['_extra_state']['model_params']
        model_params["resuming"] = model
        self.model = get_model(model_params).to(DEVICE)
        self.wrapper = ModelWrapper(self.model)  # inference only
        if JIT:
            self.wrapper = torch.jit.script(self.wrapper)
        self.wrapper.load_state_dict(state_dict)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: np.ndarray,
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        coord = torch.from_numpy(coord)
        cell = torch.from_numpy(cell)
        atype = torch.from_numpy(atype)
        model_pred = infer_model(self.model, coord, cell, atype)
        e = model_pred["energy"].cpu().detach().numpy()
        f = model_pred["force"].cpu().detach().numpy()
        v = model_pred["virial"].cpu().detach().numpy()
        return e, f, v