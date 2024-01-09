from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from dflow.python import OP, OPIO, Artifact, NestedDict, OPIOSign, Parameter


class RunTrain(OP, ABC):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "train_systems": Artifact(List[Path]),
                "valid_systems": Artifact(List[Path]),
                "pretrained_model": Artifact(Path, optional=True),
                "model": Artifact(Path, optional=True),
                "train_params": dict,
                "finetune_args": Parameter(str, default=""),
                "old_systems": Artifact(List[Path], optional=True),
                "old_ratio": Parameter(float, default=0.0),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "model": Artifact(Path),
                "output_files": Artifact(List[Path]),
            }
        )

    @abstractmethod
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        pass
