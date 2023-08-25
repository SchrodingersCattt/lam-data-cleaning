import glob
import logging
import os
from copy import deepcopy
from importlib import import_module
from typing import List, Union

import dflow
import dpclean
from dflow import (InputArtifact, InputParameter, OutputParameter, S3Artifact,
                   Step, Steps, Workflow, argo_range, if_expression,
                   upload_artifact)
from dflow.plugins.datasets import DatasetsArtifact
from dflow.plugins.dispatcher import DispatcherExecutor, update_dict
from dflow.python import PythonOPTemplate, Slices
from dpclean.op import SplitDataset, Summary


class ActiveLearning(Steps):
    def __init__(self, select_op, train_op, select_image, train_image,
                 select_image_pull_policy=None, train_image_pull_policy=None,
                 select_executor=None, train_executor=None, resume=True,
                 resume_train_params=None, finetune_args=""):
        super().__init__("active-learning-loop")
        self.inputs.parameters["iter"] = InputParameter(value=0, type=int)
        self.inputs.parameters["max_selected"] = InputParameter(type=Union[int, List[int]])
        self.inputs.parameters["threshold"] = InputParameter(type=float)
        self.inputs.parameters["train_params"] = InputParameter(type=dict)
        self.inputs.parameters["learning_curve"] = InputParameter(type=dict)
        self.inputs.parameters["select_type"] = InputParameter(type=str)
        self.inputs.parameters["ratio_selected"] = InputParameter(type=Union[float, List[float]])
        self.inputs.artifacts["candidate_systems"] = InputArtifact()
        self.inputs.artifacts["current_systems"] = InputArtifact()
        self.inputs.artifacts["valid_systems"] = InputArtifact()
        self.inputs.artifacts["finetune_model"] = InputArtifact()
        self.inputs.artifacts["model"] = InputArtifact()
        self.outputs.parameters["learning_curve"] = OutputParameter(type=dict)

        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                      image_pull_policy=train_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={"train_params": self.inputs.parameters["train_params"],
                        "finetune_args": finetune_args},
            artifacts={"train_systems": self.inputs.artifacts["current_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "finetune_model": self.inputs.artifacts["finetune_model"],
                       "model": self.inputs.artifacts["model"]},
            executor=train_executor,
            key="iter-%s-train" % self.inputs.parameters["iter"],
        )
        self.add(train_step)

        select_step = Step(
            "select-samples",
            template=PythonOPTemplate(select_op, image=select_image,
                                      image_pull_policy=select_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={"max_selected": self.inputs.parameters["max_selected"],
                        "iter": self.inputs.parameters["iter"],
                        "threshold": self.inputs.parameters["threshold"],
                        "learning_curve": self.inputs.parameters["learning_curve"],
                        "select_type": self.inputs.parameters["select_type"],
                        "ratio_selected": self.inputs.parameters["ratio_selected"]},
            artifacts={"current_systems": self.inputs.artifacts["current_systems"],
                       "candidate_systems": self.inputs.artifacts["candidate_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "model": train_step.outputs.artifacts["model"]},
            executor=select_executor,
            key="iter-%s-select" % self.inputs.parameters["iter"],
        )
        self.add(select_step)

        if resume and resume_train_params is not None:
            train_params = resume_train_params
        else:
            train_params = self.inputs.parameters["train_params"]
        next_step = Step(
            "next-loop",
            template=self,
            parameters={"iter": self.inputs.parameters["iter"] + 1,
                        "max_selected": self.inputs.parameters["max_selected"],
                        "threshold": self.inputs.parameters["threshold"],
                        "train_params": train_params,
                        "learning_curve": select_step.outputs.parameters["learning_curve"],
                        "select_type": self.inputs.parameters["select_type"],
                        "ratio_selected": self.inputs.parameters["ratio_selected"]},
            artifacts={"candidate_systems": select_step.outputs.artifacts["remaining_systems"],
                       "current_systems": select_step.outputs.artifacts["current_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "finetune_model": self.inputs.artifacts["finetune_model"],
                       "model": train_step.outputs.artifacts["model"]
                       if resume else None},
            when="%s > 0" % select_step.outputs.parameters["n_selected"],
        )
        self.add(next_step)

        self.outputs.parameters["learning_curve"].value_from_expression = if_expression(
            _if=select_step.outputs.parameters["n_selected"] > 0,
            _then=next_step.outputs.parameters["learning_curve"],
            _else=select_step.outputs.parameters["learning_curve"])


def import_func(s : str):
    fields = s.split(".")
    if fields[0] == __name__ or fields[0] == "":
        fields[0] = ""
        mod = import_module(".".join(fields[:-1]), package=__name__)
    else:
        mod = import_module(".".join(fields[:-1]))
    return getattr(mod, fields[-1])


def get_artifact(urn, name="data", detect_systems=False):
    if urn is None:
        return None
    elif isinstance(urn, str) and urn.startswith("oss://"):
        return S3Artifact(key=urn[6:])
    elif isinstance(urn, str) and urn.startswith("launching+datasets://"):
        return DatasetsArtifact.from_urn(urn)
    else:
        if detect_systems:
            path = []
            for ds in urn if isinstance(urn, list) else [urn]:
                for f in glob.glob(os.path.join(ds, "**/type.raw"), recursive=True):
                    path.append(os.path.dirname(f))
        else:
            path = urn
        artifact = upload_artifact(path)
        if hasattr(artifact, "key"):
            logging.info("%s uploaded to %s" % (name, artifact.key))
        return artifact


def build_workflow(config):
    task = config.get("task", "active_learning")
    if task == "active_learning":
        return build_active_learning_workflow(config)
    elif task == "train_only":
        return build_train_only_workflow(config)


def build_train_only_workflow(config):
    dflow.config["detect_empty_dir"] = False
    wf_name = config.get("name", "clean-data")
    zero_shot = config.get("zero_shot", False)
    dataset = config.get("dataset", None)
    dataset_artifact = get_artifact(dataset, "dataset")
    finetune_model = config.get("finetune_model", None)
    finetune_model_artifact = get_artifact(finetune_model, "finetune model")
    valid_data = config["valid_data"]
    valid_data_artifact = get_artifact(valid_data, "validation data", True)

    stat = config.get("statistics", {})
    stat_op = import_func(stat.get("op", "dpclean.op.Statistics"))
    stat_image = stat.get("image", "dptechnology/dpdata")
    stat_image_pull_policy = stat.get("image_pull_policy")
    stat_executor = stat.get("executor")
    if stat_executor is not None:
        stat_executor = DispatcherExecutor(**stat_executor)

    train = config["train"]
    train_op = import_func(train["op"])
    train_image = train["image"]
    train_image_pull_policy = train.get("image_pull_policy")
    train_executor = train.get("executor")
    if train_executor is not None:
        train_executor = DispatcherExecutor(**train_executor)
    train_params = train["params"]
    finetune_args = train.get("finetune_args", "")

    valid = config["valid"]
    valid_op = import_func(valid["op"])
    valid_image = valid["image"]
    valid_image_pull_policy = valid.get("image_pull_policy")
    valid_executor = valid.get("executor")
    if valid_executor is not None:
        valid_executor = DispatcherExecutor(**valid_executor)

    if zero_shot:
        zero_steps = Steps("zero-shot")
        zero_params = deepcopy(train_params)
        zero_params["training"]["numb_steps"] = 1
        zero_params["training"]["disp_freq"] = 1
        zero_params["training"]["save_freq"] = 1
        zero_params["learning_rate"]["start_lr"] = 1e-50
        zero_params["learning_rate"]["stop_lr"] = 1e-50
        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                    image_pull_policy=train_image_pull_policy,
                                    python_packages=dpclean.__path__),
            parameters={"train_params": zero_params,
                        "finetune_args": finetune_args},
            artifacts={"train_systems": valid_data_artifact,
                    "valid_systems": valid_data_artifact,
                    "finetune_model": finetune_model_artifact,
                    "model": None},
            executor=train_executor,
            key="train-zero",
        )
        zero_steps.add(train_step)
        valid_step = Step(
            "validate",
            template=PythonOPTemplate(valid_op, image=valid_image,
                                    image_pull_policy=valid_image_pull_policy,
                                    python_packages=dpclean.__path__),
            artifacts={"valid_systems": valid_data_artifact,
                    "model": train_step.outputs.artifacts["model"]},
            executor=valid_executor,
            key="valid-zero",
        )
        zero_steps.add(valid_step)
        zero_steps.outputs.parameters["results"] = OutputParameter(value_from_parameter=valid_step.outputs.parameters["results"])
        zero_step = Step("zero-shot", template=zero_steps)

    steps = Steps("train-validate")
    steps.inputs.parameters["item"] = InputParameter(type=int)
    steps.inputs.artifacts["train_systems"] = InputArtifact()
    train_step = Step(
        "train",
        template=PythonOPTemplate(train_op, image=train_image,
                                  image_pull_policy=train_image_pull_policy,
                                  python_packages=dpclean.__path__),
        parameters={"train_params": train_params,
                    "finetune_args": finetune_args},
        artifacts={"train_systems": steps.inputs.artifacts["train_systems"],
                   "valid_systems": valid_data_artifact,
                   "finetune_model": finetune_model_artifact,
                   "model": None},
        executor=train_executor,
        key="train-%s" % steps.inputs.parameters["item"],
    )
    steps.add(train_step)
    valid_step = Step(
        "validate",
        template=PythonOPTemplate(valid_op, image=valid_image,
                                  image_pull_policy=valid_image_pull_policy,
                                  python_packages=dpclean.__path__),
        artifacts={"valid_systems": valid_data_artifact,
                   "model": train_step.outputs.artifacts["model"]},
        executor=valid_executor,
        key="valid-%s" % steps.inputs.parameters["item"],
    )
    steps.add(valid_step)
    steps.outputs.parameters["results"] = OutputParameter(value_from_parameter=valid_step.outputs.parameters["results"])

    wf = Workflow(wf_name, parameters={"input": config})
    if dataset_artifact is not None:
        stat_step = Step(
            "statistics",
            template=PythonOPTemplate(stat_op,
                                    image=stat_image,
                                    image_pull_policy=stat_image_pull_policy,
                                    python_packages=dpclean.__path__),
            artifacts={"dataset": dataset_artifact},
            executor=stat_executor,
            key="statistics",
        )
        wf.add(stat_step)

        train_step = Step(
            "parallel-train",
            template=steps,
            parameters={"item": "{{item}}"},
            artifacts={"train_systems": dataset_artifact},
            slices=Slices(input_artifact=["train_systems"],
                        output_parameter=["results"]),
            with_param=argo_range(stat_step.outputs.parameters["n_subsets"]),
        )
        if zero_shot:
            wf.add([train_step, zero_step])
        else:
            wf.add(train_step)
    elif zero_shot:
        wf.add(zero_step)

    parameters = {}
    if dataset_artifact is not None:
        parameters["size_list"] = stat_step.outputs.parameters["size_list"]
        parameters["results"] = train_step.outputs.parameters["results"]
    if zero_shot:
        parameters["zero_result"] = zero_step.outputs.parameters["results"]
    sum_step = Step(
        "summary",
        template=PythonOPTemplate(Summary,
                                  image="dptechnology/dpdata",
                                  image_pull_policy="IfNotPresent",
                                  python_packages=dpclean.__path__),
        parameters=parameters,
        key="summary",
    )
    wf.add(sum_step)
    return wf

def build_active_learning_workflow(config):
    dflow.config["detect_empty_dir"] = False
    wf_name = config.get("name", "clean-data")
    dataset = config["dataset"]
    init_data = config.get("init_data", None)
    n_init = config.get("n_init", None)
    ratio_init = config.get("ratio_init", None)
    select_type = config.get("select_type", "global")
    valid_data = config["valid_data"]
    finetune_model = config.get("finetune_model", None)
    resume = config.get("resume", True)
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
    max_selected = select.get("max_selected", None)
    ratio_selected = select.get("ratio_selected", None)
    threshold = select["threshold"]

    train_op = import_func(train["op"])
    train_image = train["image"]
    train_image_pull_policy = train.get("image_pull_policy")
    train_executor = train.get("executor")
    if train_executor is not None:
        train_executor = DispatcherExecutor(**train_executor)
    train_params = train["params"]
    resume_train_params = None
    resume_params = train.get("resume_params", None) if resume else None
    if resume_params is not None:
        resume_train_params = deepcopy(train_params)
        update_dict(resume_train_params, resume_params)
    finetune_args = train.get("finetune_args", "")

    wf = Workflow(wf_name, parameters={"input": config})
    dataset_artifact = get_artifact(dataset, "dataset")
    init_data_artifact = get_artifact(init_data, "init data", True)
    split_step = Step(
        "split-dataset",
        template=PythonOPTemplate(split_op, image=split_image,
                                  image_pull_policy=split_image_pull_policy,
                                  python_packages=dpclean.__path__),
        parameters={"n_init": n_init if init_data is None else 0,
                    "ratio_init": ratio_init if init_data is None else 0.0,
                    "select_type": select_type},
        artifacts={"dataset": dataset_artifact,
                   "init_systems": init_data_artifact},
        executor=split_executor,
        key="split-dataset"
    )
    wf.add(split_step)

    active_learning = ActiveLearning(
        select_op, train_op, select_image, train_image,
        select_image_pull_policy, train_image_pull_policy, select_executor,
        train_executor, resume, resume_train_params, finetune_args)

    finetune_model_artifact = get_artifact(finetune_model, "finetune model")
    valid_data_artifact = get_artifact(valid_data, "validation data", True)
    loop_step = Step(
        "active-learning-loop",
        template=active_learning,
        parameters={"max_selected": max_selected,
                    "threshold": threshold,
                    "train_params": train_params,
                    "learning_curve": {},
                    "select_type": select_type,
                    "ratio_selected": ratio_selected},
        artifacts={"current_systems": split_step.outputs.artifacts["init_systems"],
                   "candidate_systems": split_step.outputs.artifacts["systems"],
                   "valid_systems": valid_data_artifact,
                   "finetune_model": finetune_model_artifact,
                   "model": None})
    wf.add(loop_step)
    return wf
