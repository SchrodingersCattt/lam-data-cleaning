import argparse
import json
import logging
import os
from typing import List, Optional

from dflow import Workflow
from dpclean.flow import build_workflow


log_level = os.environ.get('LOG_LEVEL', 'INFO')
if log_level:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))


def main_parser():
    parser = argparse.ArgumentParser(
        description="Data cleaning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Valid subcommands", dest="command")

    parser_submit = subparsers.add_parser(
        "submit",
        help="Submit a data-cleaning workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_submit.add_argument("CONFIG", help="the config file.")

    parser_resubmit = subparsers.add_parser(
        "resubmit",
        help="Resubmit a data-cleaning workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_resubmit.add_argument("CONFIG", help="the config file.")
    parser_resubmit.add_argument("ID", help="the workflow ID.")
    parser_resubmit.add_argument(
        "-f",
        "--from",
        type=str,
        default=None,
        help="key of the step restart from",
    )
    return parser


def parse_args(args: Optional[List[str]] = None):
    """Commandline options argument parsing.

    Parameters
    ----------
    args : List[str]
        list of command line arguments, main purpose is testing default option
        None takes arguments from sys.argv
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    return parsed_args


def main():
    args = parse_args()
    if args.command == "submit":
        with open(args.CONFIG, "r") as f:
            config = json.load(f)
        wf = build_workflow(config)
        wf.submit()
    elif args.command == "resubmit":
        wf0 = Workflow(id=args.ID)
        reused_steps = [step for step in wf0.query_step() if step.key is not None and step.phase == "Succeeded"]
        restart_from = getattr(args, "from", None)
        if restart_from is not None:
            fields = restart_from.split("-")
            iter = int(fields[1])
            task = fields[2]
            if task == "train":
                reused_steps = [s for s in reused_steps
                                if (s.key.startswith("iter-") and int(s.key.split("-")[1]) < iter)
                                or not s.key.startswith("iter-")]
            elif task == "select":
                reused_steps = [s for s in reused_steps
                                if (s.key.startswith("iter-") and int(s.key.split("-")[1]) < iter)
                                or s.key == "iter-%s-train" % iter or not s.key.startswith("iter-")]
        with open(args.CONFIG, "r") as f:
            config = json.load(f)
        wf = build_workflow(config)
        wf.submit(reuse_step=reused_steps)


if __name__ == "__main__":
    main()
