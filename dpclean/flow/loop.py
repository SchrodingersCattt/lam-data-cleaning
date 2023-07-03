import logging
from importlib import import_module

import dpclean
from dflow import (InputArtifact, InputParameter, S3Artifact, Step, Steps,
                   Workflow, upload_artifact)
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow.python import PythonOPTemplate
from dpclean.op import SplitDataset


class ActiveLearning(Steps):
    def __init__(self, select_op, train_op, select_image, train_image,
                 select_image_pull_policy=None, train_image_pull_policy=None,
                 select_executor=None, train_executor=None):
        super().__init__("active-learning-loop")
        self.inputs.parameters["iter"] = InputParameter(value=0, type=int)
        self.inputs.parameters["max_selected"] = InputParameter(type=int)
        self.inputs.parameters["threshold"] = InputParameter(type=float)
        self.inputs.parameters["train_params"] = InputParameter(type=dict)
        self.inputs.artifacts["candidate_systems"] = InputArtifact()
        self.inputs.artifacts["current_systems"] = InputArtifact()
        self.inputs.artifacts["valid_systems"] = InputArtifact()
        self.inputs.artifacts["model"] = InputArtifact()

        select_step = Step(
            "select-samples",
            template=PythonOPTemplate(select_op, image=select_image,
                                      python_packages=dpclean.__path__),
            parameters={"max_selected": self.inputs.parameters["max_selected"],
                        "threshold": self.inputs.parameters["threshold"]},
            artifacts={"systems": self.inputs.artifacts["candidate_systems"],
                       "model": self.inputs.artifacts["model"]},
            executor=select_executor,
            key="iter-%s-select" % self.inputs.parameters["iter"],
        )
        self.add(select_step)

        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                      python_packages=dpclean.__path__),
            parameters={"train_params": self.inputs.parameters["train_params"]},
            artifacts={"current_systems": self.inputs.artifacts["current_systems"],
                       "added_systems": select_step.outputs.artifacts["selected_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "model": self.inputs.artifacts["model"]},
            when="%s > 0" % select_step.outputs.parameters["n_selected"],
            executor=train_executor,
            key="iter-%s-train" % self.inputs.parameters["iter"],
        )
        self.add(train_step)

        next_step = Step(
            "next-loop",
            template=self,
            parameters={"iter": self.inputs.parameters["iter"] + 1,
                        "max_selected": self.inputs.parameters["max_selected"],
                        "threshold": self.inputs.parameters["threshold"],
                        "train_params": self.inputs.parameters["train_params"]},
            artifacts={"candidate_systems": select_step.outputs.artifacts["remaining_systems"],
                       "current_systems": train_step.outputs.artifacts["current_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "model": train_step.outputs.artifacts["model"]},
            when="%s > 0" % select_step.outputs.parameters["n_selected"],
        )
        self.add(next_step)


def import_func(s : str):
    fields = s.split(".")
    if fields[0] == __name__ or fields[0] == "":
        fields[0] = ""
        mod = import_module(".".join(fields[:-1]), package=__name__)
    else:
        mod = import_module(".".join(fields[:-1]))
    return getattr(mod, fields[-1])


def build_workflow(config):
    dataset = config["dataset"]
    init_data = config.get("init_data", [])
    valid_data = config["valid_data"]
    model = config["model"]
    split = config.get("split", {})
    select = config["select"]
    train = config["train"]

    split_op = split.get("op")
    if split_op is None:
        split_op = SplitDataset
    else:
        split_op = import_func(split_op)
    split_image = split.get("image", "dptechnology/dpdata")
    split_image_pull_policy = split.get("image_pull_policy")
    split_executor = split.get("executor")
    if split_executor is not None:
        split_executor = DispatcherExecutor(**split_executor)

    select_op = import_func(select["op"])
    select_image = select["image"]
    select_image_pull_policy = select.get("image_pull_policy")
    select_executor = select.get("executor")
    if select_executor is not None:
        select_executor = DispatcherExecutor(**select_executor)
    max_selected = select["max_selected"]
    threshold = select["threshold"]

    train_op = import_func(train["op"])
    train_image = train["image"]
    train_image_pull_policy = train.get("image_pull_policy")
    train_executor = train.get("executor")
    if train_executor is not None:
        train_executor = DispatcherExecutor(**train_executor)
    train_params = train["params"]

    wf = Workflow("clean-data")
    if isinstance(dataset, str) and dataset.startswith("oss://"):
        dataset_artifact = S3Artifact(key=dataset[6:])
    else:
        dataset_artifact = upload_artifact(dataset)
        logging.info("Dataset uploaded to %s" % dataset_artifact.key)
    split_step = Step(
        "split-dataset",
        template=PythonOPTemplate(split_op, image=split_image,
                                  image_pull_policy=split_image_pull_policy,
                                  python_packages=dpclean.__path__),
        artifacts={"dataset": dataset_artifact},
        executor=split_executor,
        key="split-dataset"
    )
    wf.add(split_step)

    active_learning = ActiveLearning(select_op, train_op, select_image,
                                     train_image, select_image_pull_policy,
                                     train_image_pull_policy, select_executor,
                                     train_executor)

    if isinstance(model, str) and model.startswith("oss://"):
        model_artifact = S3Artifact(key=model[6:])
    else:
        model_artifact = upload_artifact(model)
        logging.info("Model uploaded to %s" % model_artifact.key)
    if isinstance(init_data, str) and init_data.startswith("oss://"):
        init_data_artifact = S3Artifact(key=init_data[6:])
    else:
        init_data_artifact = upload_artifact(init_data)
        logging.info("Init data uploaded to %s" % init_data_artifact.key)
    if isinstance(valid_data, str) and valid_data.startswith("oss://"):
        valid_data_artifact = S3Artifact(key=valid_data[6:])
    else:
        valid_data_artifact = upload_artifact(valid_data)
        logging.info("Validation data uploaded to %s" % valid_data_artifact.key)
    loop_step = Step(
        "active-learning-loop",
        template=active_learning,
        parameters={"max_selected": max_selected,
                    "threshold": threshold,
                    "train_params": train_params},
        artifacts={"current_systems": init_data_artifact,
                   "candidate_systems": split_step.outputs.artifacts["systems"],
                   "valid_systems": valid_data_artifact,
                   "model": model_artifact},
    )
    wf.add(loop_step)
    return wf
