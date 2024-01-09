# lam-data-cleaning

This project is to reduce the number of samples required for training a large atomic model (LAM) developed based on the cloud-native workflow framework [dflow](https://github.com/deepmodeling/dflow). So far, [DeePMD](https://github.com/deepmodeling/deepmd-kit), [OCP](https://github.com/Open-Catalyst-Project/ocp) and [Nequip](https://github.com/mir-group/nequip) are supported.

## Installation

Install this project by

```
pip install .
```

Configure workflow server required by dflow before submit a workflow by command line

```
dpclean submit input.json
```

Some example input files can be found in [examples](examples/).