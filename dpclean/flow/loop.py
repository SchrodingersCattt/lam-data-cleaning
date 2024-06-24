import glob
import logging
import os
from copy import deepcopy
from importlib import import_module
from typing import List

import dflow
import dpclean
from dflow import (InputArtifact, InputParameter, OutputParameter, S3Artifact,
                   Step, Steps, Workflow, argo_enumerate, if_expression,
                   upload_artifact)
from dflow.plugins.datasets import DatasetsArtifact
from dflow.plugins.dispatcher import DispatcherExecutor, update_dict
from dflow.python import PythonOPTemplate, Slices, upload_packages
from dpclean.op import Merge, SplitDataset, Summary


class ActiveLearning(Steps):
    def __init__(self, select_op, train_op, select_image, train_image,
                 select_image_pull_policy=None, train_image_pull_policy=None,
                 select_executor=None, train_executor=None, resume=True,
                 resume_train_params=None, finetune_args="", max_iter=None,
                 train_optional_args=None, select_optional_args=None):
        super().__init__("active-learning-loop")
        self.inputs.parameters["iter"] = InputParameter(value=0, type=int)
        self.inputs.parameters["train_params"] = InputParameter(type=dict)
        self.inputs.parameters["learning_curve"] = InputParameter(type=dict)
        self.inputs.parameters["select_type"] = InputParameter(type=str)
        self.inputs.parameters["ratio_selected"] = InputParameter(type=List[float])
        self.inputs.parameters["batch_size"] = InputParameter(type=str, value="auto")
        self.inputs.artifacts["candidate_systems"] = InputArtifact()
        self.inputs.artifacts["current_systems"] = InputArtifact()
        self.inputs.artifacts["valid_systems"] = InputArtifact()
        self.inputs.artifacts["pretrained_model"] = InputArtifact()
        self.inputs.artifacts["model"] = InputArtifact()
        self.outputs.parameters["learning_curve"] = OutputParameter(type=dict)

        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                      image_pull_policy=train_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={"train_params": self.inputs.parameters["train_params"],
                        "finetune_args": finetune_args,
                        "optional_args": train_optional_args or {},},
            artifacts={"train_systems": self.inputs.artifacts["current_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "pretrained_model": self.inputs.artifacts["pretrained_model"],
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
            parameters={"iter": self.inputs.parameters["iter"],
                        "learning_curve": self.inputs.parameters["learning_curve"],
                        "select_type": self.inputs.parameters["select_type"],
                        "ratio_selected": self.inputs.parameters["ratio_selected"],
                        "train_params": self.inputs.parameters["train_params"],
                        "batch_size": self.inputs.parameters["batch_size"],
                        "optional_args": select_optional_args or {},},
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
                        "train_params": train_params,
                        "learning_curve": select_step.outputs.parameters["learning_curve"],
                        "select_type": self.inputs.parameters["select_type"],
                        "ratio_selected": self.inputs.parameters["ratio_selected"],
                        "batch_size": self.inputs.parameters["batch_size"]},
            artifacts={"candidate_systems": select_step.outputs.artifacts["remaining_systems"],
                       "current_systems": select_step.outputs.artifacts["current_systems"],
                       "valid_systems": self.inputs.artifacts["valid_systems"],
                       "pretrained_model": self.inputs.artifacts["pretrained_model"],
                       "model": train_step.outputs.artifacts["model"]
                       if resume else None},
            when="%s < %s" % (self.inputs.parameters["iter"], max_iter),
        )
        self.add(next_step)

        self.outputs.parameters["learning_curve"].value_from_expression = if_expression(
            _if=self.inputs.parameters["iter"] < max_iter,
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
    upload_packages.extend(config.get("upload_packages", []))
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
    if isinstance(dataset, list):
        for i, subset in enumerate(dataset):
            path_list = []
            for ds in subset if isinstance(subset, list) else [subset]:
                for f in glob.glob(os.path.join(ds, "**/type.raw"), recursive=True):
                    path_list.append(os.path.dirname(f))
            dataset[i] = path_list
    dataset_artifact = get_artifact(dataset, "dataset")
    pretrained_model = config.get("pretrained_model", config.get("finetune_model"))
    pretrained_model_artifact = get_artifact(pretrained_model, "finetune model")
    valid_data = config["valid_data"]
    valid_data_artifact = get_artifact(valid_data, "validation data", True)
    old_data = config.get("old_data", None)
    old_data_artifact = get_artifact(old_data, "old data", True)
    old_ratio = config.get("old_ratio", 0.0)

    stat = config.get("statistics", {})
    stat_op = import_func(stat.get("op", "dpclean.op.Statistics"))
    stat_image = stat.get("image", "dptechnology/dpdata")
    stat_image_pull_policy = stat.get("image_pull_policy")
    stat_executor = stat.get("executor")
    if stat_executor is not None:
        stat_executor = DispatcherExecutor(**stat_executor)

    summ = config.get("summary", {})
    summ_image = summ.get("image", "dptechnology/dpdata")

    train = config["train"]
    train_op = import_func(train["op"])
    train_image = train["image"]
    train_image_pull_policy = train.get("image_pull_policy")
    train_executor = train.get("executor")
    if train_executor is not None:
        train_executor = DispatcherExecutor(**train_executor)
    train_params = train["params"]
    finetune_args = train.get("finetune_args", "")
    train_optional_args = train.get("optional_args", {})

    valid = config["valid"]
    batch_size = valid.get("batch_size", "auto")
    valid_op = import_func(valid["op"])
    valid_image = valid["image"]
    valid_image_pull_policy = valid.get("image_pull_policy")
    valid_executor = valid.get("executor")
    if valid_executor is not None:
        valid_executor = DispatcherExecutor(**valid_executor)
    valid_optional_args = valid.get("optional_args", {})

    if zero_shot:
        zero_steps = Steps("zero-shot")
        zero_params = deepcopy(train_params)
        zero_params["training"]["numb_steps"] = 1
        zero_params["training"]["disp_freq"] = 1
        zero_params["training"]["save_freq"] = 1
        zero_params["learning_rate"]["start_lr"] = 1e-10
        zero_params["learning_rate"]["stop_lr"] = 1e-10
        zero_params["learning_rate"]["decay_steps"] = 1
        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                    image_pull_policy=train_image_pull_policy,
                                    python_packages=dpclean.__path__),
            parameters={
                "train_params": zero_params,
                "finetune_args": finetune_args,
                "old_ratio": old_ratio,
                "optional_args": train_optional_args,
            },
            artifacts={
                "train_systems": valid_data_artifact,
                "valid_systems": valid_data_artifact,
                "pretrained_model": pretrained_model_artifact,
                "model": None,
                "old_systems": old_data_artifact,
            },
            executor=train_executor,
            key="train-zero",
        )
        zero_steps.add(train_step)
        valid_step = Step(
            "validate",
            template=PythonOPTemplate(valid_op, image=valid_image,
                                    image_pull_policy=valid_image_pull_policy,
                                    python_packages=dpclean.__path__),
            parameters={"train_params": zero_params,
                        "batch_size": batch_size,
                        "optional_args": valid_optional_args,},
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
    steps.inputs.parameters["size"] = InputParameter(type=int)
    steps.inputs.artifacts["train_systems"] = InputArtifact()
    train_step = Step(
        "train",
        template=PythonOPTemplate(train_op, image=train_image,
                                  image_pull_policy=train_image_pull_policy,
                                  python_packages=dpclean.__path__),
        parameters={
            "train_params": train_params,
            "finetune_args": finetune_args,
            "old_ratio": old_ratio,
            "optional_args": train_optional_args,
        },
        artifacts={
            "train_systems": steps.inputs.artifacts["train_systems"],
            "valid_systems": valid_data_artifact,
            "pretrained_model": pretrained_model_artifact,
            "model": None,
            "old_systems": old_data_artifact,
        },
        executor=train_executor,
        key="train-%s" % steps.inputs.parameters["item"],
    )
    steps.add(train_step)
    valid_step = Step(
        "validate",
        template=PythonOPTemplate(valid_op, image=valid_image,
                                  image_pull_policy=valid_image_pull_policy,
                                  python_packages=dpclean.__path__),
        parameters={"train_params": train_params,
                    "batch_size": batch_size,
                    "optional_args": valid_optional_args,},
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
            parameters={"item": "{{item.order}}",
                        "size": "{{item.value}}"},
            artifacts={"train_systems": dataset_artifact},
            slices=Slices("{{item.order}}",
                          input_artifact=["train_systems"],
                          output_parameter=["results"]),
            with_param=argo_enumerate(stat_step.outputs.parameters["size_list"]),
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
                                  image=summ_image,
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
    zero_shot = config.get("zero_shot", False)
    dataset = config["dataset"]
    ratio_init = config.get("ratio_init", None)
    select_type = config.get("select_type", "global")
    valid_data = config["valid_data"]
    pretrained_model = config.get("pretrained_model", config.get("pretrained_model"))
    resume = config.get("resume", False)
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
    ratio_selected = select["ratio_selected"]
    batch_size = select.get("batch_size", "auto")
    select_optional_args = select.get("optional_args", {})

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
    train_optional_args = train.get("optional_args", {})

    wf = Workflow(wf_name, parameters={"input": config})
    dataset_artifact = get_artifact(dataset, "dataset", True)

    pretrained_model_artifact = get_artifact(pretrained_model, "finetune model")
    valid_data_artifact = get_artifact(valid_data, "validation data", True)
    parallel_steps = []
    train_all = len(ratio_selected) > 0 and ratio_selected[-1] == 1.0 and not resume
    if train_all:
        ratio_selected.pop()
        all_steps = Steps("all")
        all_steps.inputs.artifacts["train_systems"] = InputArtifact()
        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                      image_pull_policy=train_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={
                "train_params": train_params,
                "finetune_args": finetune_args,
                "optional_args": train_optional_args,
            },
            artifacts={
                "train_systems": all_steps.inputs.artifacts["train_systems"],
                "valid_systems": valid_data_artifact,
                "pretrained_model": pretrained_model_artifact,
                "model": None,
            },
            executor=train_executor,
            key="train-all",
        )
        all_steps.add(train_step)
        valid_step = Step(
            "validate",
            template=PythonOPTemplate(select_op, image=select_image,
                                      image_pull_policy=select_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={"iter": 0,
                        "learning_curve": {},
                        "ratio_selected": [0.0],
                        "train_params": train_params,
                        "batch_size": batch_size,
                        "optional_args": select_optional_args,},
            artifacts={"current_systems": all_steps.inputs.artifacts["train_systems"],
                       "candidate_systems": upload_artifact([]),
                       "valid_systems": valid_data_artifact,
                       "model": train_step.outputs.artifacts["model"]},
            executor=select_executor,
            key="valid-all",
        )
        all_steps.add(valid_step)
        all_steps.outputs.parameters["learning_curve"] = OutputParameter(value_from_parameter=valid_step.outputs.parameters["learning_curve"])
        all_step = Step(
            "all",
            template=all_steps,
            artifacts={
                "train_systems": dataset_artifact},
        )
        parallel_steps.append(all_step)

    def get_zero_steps(ratio_selected, candidate_systems):
        zero_params = deepcopy(train_params)
        zero_params["training"]["numb_steps"] = 1
        zero_params["training"]["disp_freq"] = 1
        zero_params["training"]["save_freq"] = 1
        zero_params["learning_rate"]["start_lr"] = 1e-10
        zero_params["learning_rate"]["stop_lr"] = 1e-10
        zero_params["learning_rate"]["decay_steps"] = 1
        train_step = Step(
            "train",
            template=PythonOPTemplate(train_op, image=train_image,
                                      image_pull_policy=train_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={
                "train_params": zero_params,
                "finetune_args": finetune_args,
                "optional_args": train_optional_args,
            },
            artifacts={
                "train_systems": valid_data_artifact,
                "valid_systems": valid_data_artifact,
                "pretrained_model": pretrained_model_artifact,
                "model": None,
            },
            executor=train_executor,
            key="train-zero",
        )
        valid_step = Step(
            "validate",
            template=PythonOPTemplate(select_op, image=select_image,
                                      image_pull_policy=select_image_pull_policy,
                                      python_packages=dpclean.__path__),
            parameters={"iter": 0,
                        "learning_curve": {},
                        "ratio_selected": ratio_selected,
                        "train_params": zero_params,
                        "batch_size": batch_size,
                        "optional_args": select_optional_args,},
            artifacts={"current_systems": upload_artifact([]),
                       "candidate_systems": candidate_systems,
                       "valid_systems": valid_data_artifact,
                       "model": train_step.outputs.artifacts["model"]},
            executor=select_executor,
            key="valid-zero",
        )
        return train_step, valid_step


    merge = config.get("merge", {})
    merge_image = merge.get("image", "dptechnology/dpdata")
    
    active_learning_steps = Steps("active-learning")
    active_learning = ActiveLearning(
        select_op, train_op, select_image, train_image,
        select_image_pull_policy, train_image_pull_policy, select_executor,
        train_executor, resume, resume_train_params, finetune_args, len(ratio_selected),
        train_optional_args, select_optional_args)
    if ratio_init > 0:
        split_step = Step(
            "split-dataset",
            template=PythonOPTemplate(split_op, image=split_image,
                                    image_pull_policy=split_image_pull_policy,
                                    python_packages=dpclean.__path__),
            parameters={"ratio_init": ratio_init},
            artifacts={"dataset": dataset_artifact},
            executor=split_executor,
            key="split-dataset"
        )
        active_learning_steps.add(split_step)

        loop_step = Step(
            "active-learning-loop",
            template=active_learning,
            parameters={"train_params": train_params,
                        "learning_curve": {},
                        "select_type": select_type,
                        "ratio_selected": ratio_selected,
                        "batch_size": batch_size},
            artifacts={"current_systems": split_step.outputs.artifacts["init_systems"],
                       "candidate_systems": split_step.outputs.artifacts["systems"],
                       "valid_systems": valid_data_artifact,
                       "pretrained_model": pretrained_model_artifact,
                       "model": None})
        active_learning_steps.add(loop_step)
        active_learning_steps.outputs.parameters["learning_curve"] = OutputParameter(value_from_parameter=loop_step.outputs.parameters["learning_curve"])
    else:
        zero_shot = False
        train_step, valid_step = get_zero_steps(ratio_selected, dataset_artifact)
        active_learning_steps.add(train_step)
        active_learning_steps.add(valid_step)
        if len(ratio_selected) > 0:
            loop_step = Step(
                "active-learning-loop",
                template=active_learning,
                parameters={"iter": 1,
                            "train_params": train_params,
                            "learning_curve": valid_step.outputs.parameters["learning_curve"],
                            "select_type": select_type,
                            "ratio_selected": ratio_selected,
                            "batch_size": batch_size},
                artifacts={"current_systems": valid_step.outputs.artifacts["current_systems"],
                           "candidate_systems": valid_step.outputs.artifacts["remaining_systems"],
                           "valid_systems": valid_data_artifact,
                           "pretrained_model": pretrained_model_artifact,
                           "model": train_step.outputs.artifacts["model"] if resume else None})
            active_learning_steps.add(loop_step)
            active_learning_steps.outputs.parameters["learning_curve"] = OutputParameter(value_from_parameter=loop_step.outputs.parameters["learning_curve"])
        else:
            active_learning_steps.outputs.parameters["learning_curve"] = OutputParameter(value_from_parameter=valid_step.outputs.parameters["learning_curve"])

    active_learning_step = Step("active-learning", active_learning_steps)
    parallel_steps.append(active_learning_step)

    if zero_shot:
        zero_steps = Steps("zero-shot")
        train_step, valid_step = get_zero_steps([], upload_artifact([]))
        zero_steps.add(train_step)
        zero_steps.add(valid_step)
        zero_steps.outputs.parameters["learning_curve"] = OutputParameter(value_from_parameter=valid_step.outputs.parameters["learning_curve"])
        zero_step = Step("zero-shot", template=zero_steps)
        parallel_steps.append(zero_step)

    wf.add(parallel_steps)
    merge_step = Step(
        "merge",
        template=PythonOPTemplate(Merge,
                                  image=merge_image,
                                  image_pull_policy="IfNotPresent",
                                  python_packages=dpclean.__path__),
        parameters={
            "loop_lcurve": active_learning_step.outputs.parameters["learning_curve"],
            "zero_lcurve": zero_step.outputs.parameters["learning_curve"] if zero_shot else {},
            "all_lcurve": all_step.outputs.parameters["learning_curve"] if train_all else {},
        },
        key="merge",
    )
    wf.add(merge_step)
    return wf
