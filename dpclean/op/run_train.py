from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from dflow.python import OP, OPIO, Artifact, OPIOSign


class RunTrain(OP, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "train_systems": Artifact(List[Path]),
                "valid_systems": Artifact(List[Path]),
                "finetune_model": Artifact(Path, optional=True),
                "model": Artifact(Path, optional=True),
                "train_params": dict,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "model": Artifact(Path),
                "output_dir": Artifact(Path),
            }
        )

    @abstractmethod
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        pass
