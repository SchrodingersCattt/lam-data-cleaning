from pathlib import Path
from typing import List, Tuple

import numpy as np
from dpclean.op import SelectSamples


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
        model_pred, _, _ = self.wrapper(coord=coord, atype=atype, box=cell)
        e, f, v = model_pred["energy"], model_pred["force"], model_pred["virial"]
        return e, f, v
