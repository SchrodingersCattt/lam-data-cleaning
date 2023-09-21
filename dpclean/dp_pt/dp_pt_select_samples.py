from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from dpclean.op import SelectSamples, Validate


class DPPTValidate(Validate):
    def load_model(self, model: Path):
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


class DPPTSelectSamples(SelectSamples, DPPTValidate):
    pass
