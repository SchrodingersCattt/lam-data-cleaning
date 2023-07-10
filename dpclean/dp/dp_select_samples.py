from pathlib import Path
from typing import List, Tuple

import numpy as np
from dpclean.op import SelectSamples, Validate


class DPValidate(Validate):
    def load_model(self, model: Path):
        from deepmd.infer import DeepPot
        self.dp = DeepPot(model)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: np.ndarray,
                 atype: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coord = coord.reshape([1, -1])
        cell = cell.reshape([1, -1])
        e, f, v = self.dp.eval(coord, cell, atype)
        return e[0], f[0], v[0]


class DPSelectSamples(SelectSamples, DPValidate):
    pass
