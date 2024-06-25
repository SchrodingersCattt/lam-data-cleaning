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

## Brief Introduction 
In `Parallel Mode`, training sets are ramdomly splitted into several groups based on the appointed ratios; In `Active Learning Mode`, training sets are initially splitted into a subset (0.001 by default), after training on the subset, `dp test` will be conducted to the remaining systems, among which with the largest error will be added to training set for the next iteration of training.

Please note that `numb_steps` and `decay_steps` should vary according to the frames, so that the epochs of every loop are controlled to be the same. For instance, if a training system contains 10000 frames and the expected training steps for full-size system is 1000000, the parameter of `numb_steps` should set as` "math.ceil(100*n)"`, `n` represents the frame number of every iteration, the coefficient `100` represents `full-size training step / full-size frame number`.
