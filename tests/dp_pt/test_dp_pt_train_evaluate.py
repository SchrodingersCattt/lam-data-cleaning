import shutil
import unittest
from pathlib import Path

import numpy as np
from dpclean.dp_pt import DPPTSelectSamples, RunDPPTTrain


class TestTrainEvaluate:
    def setUp(self):
        self.train_systems = [Path(__file__).parent.parent / "water/data/single"]
        self.valid_systems = [Path(__file__).parent.parent / "water/data/single"]
        self.coord = np.load(Path(__file__).parent.parent / "water/data/single/set.000/coord.npy").reshape([-1, 3])
        self.cell = 12.444661 * np.eye(3)
        self.atype = np.array([7] * 64 + [0] * 128)


class TestDPPTTrainEvaluate(TestTrainEvaluate, unittest.TestCase):
    def test(self):
        train_params = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": 80,
                    "rcut_smth": 2.0,
                    "rcut": 9.0,
                    "neuron": [
                        25,
                        50,
                        100
                    ],
                    "resnet_dt": False,
                    "axis_neuron": 12,
                    "seed": 1,
                    "attn": 128,
                    "attn_layer": 0,
                    "attn_dotr": True,
                    "attn_mask": False,
                    "post_ln": True,
                    "ffn": False,
                    "ffn_embed_dim": 1024,
                    "activation": "tanh",
                    "scaling_factor": 1.0,
                    "head_num": 1,
                    "normalize": True,
                    "temperature": 1.0
                },
                "fitting_net": {
                    "neuron": [
                        240,
                        240,
                        240
                    ],
                    "resnet_dt": True,
                    "seed": 1
                },
                "type_map": [
                    "H",
                    "He",
                    "Li",
                    "Be",
                    "B",
                    "C",
                    "N",
                    "O",
                    "F",
                    "Ne",
                    "Na",
                    "Mg",
                    "Al",
                    "Si",
                    "P",
                    "S",
                    "Cl",
                    "Ar",
                    "K",
                    "Ca",
                    "Sc",
                    "Ti",
                    "V",
                    "Cr",
                    "Mn",
                    "Fe",
                    "Co",
                    "Ni",
                    "Cu",
                    "Zn",
                    "Ga",
                    "Ge",
                    "As",
                    "Se",
                    "Br",
                    "Kr",
                    "Rb",
                    "Sr",
                    "Y",
                    "Zr",
                    "Nb",
                    "Mo",
                    "Tc",
                    "Ru",
                    "Rh",
                    "Pd",
                    "Ag",
                    "Cd",
                    "In",
                    "Sn",
                    "Sb",
                    "Te",
                    "I",
                    "Xe",
                    "Cs",
                    "Ba",
                    "La",
                    "Ce",
                    "Pr",
                    "Nd",
                    "Pm",
                    "Sm",
                    "Eu",
                    "Gd",
                    "Tb",
                    "Dy",
                    "Ho",
                    "Er",
                    "Tm",
                    "Yb",
                    "Lu",
                    "Hf",
                    "Ta",
                    "W",
                    "Re",
                    "Os",
                    "Ir",
                    "Pt",
                    "Au",
                    "Hg",
                    "Tl",
                    "Pb",
                    "Bi",
                    "Po",
                    "At",
                    "Rn",
                    "Fr",
                    "Ra",
                    "Ac",
                    "Th",
                    "Pa",
                    "U",
                    "Np",
                    "Pu",
                    "Am",
                    "Cm",
                    "Bk",
                    "Cf",
                    "Es",
                    "Fm",
                    "Md",
                    "No",
                    "Lr",
                    "Rf",
                    "Db",
                    "Sg",
                    "Bh",
                    "Hs",
                    "Mt",
                    "Ds",
                    "Rg",
                    "Cn",
                    "Nh",
                    "Fl",
                    "Mc",
                    "Lv",
                    "Ts",
                    "Og"
                ]
            },
            "learning_rate": {
                "type": "exp",
                "start_lr": 0.001,
                "decay_steps": 5000,
                "stop_lr": 3.51e-08
            },
            "loss": {
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0
            },
            "training": {
                "training_data": {
                    "batch_size": 1
                },
                "validation_data": {
                    "batch_size": 1
                },
                "numb_steps": 2,
                "seed": 1,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 1
            }
        }        
        op_in = {
            "train_systems": self.train_systems,
            "valid_systems": self.valid_systems,
            "train_params": train_params,
        }
        op = RunDPPTTrain()
        op_out = op.execute(op_in)
        assert op_out["model"].exists()
        assert op_out["output_dir"].is_dir()

        op = DPPTSelectSamples()
        op.load_model(op_out["model"])
        e, f, v = op.evaluate(self.coord, self.cell, self.atype)
        assert f.shape == (192, 3)

        shutil.rmtree(op_out["output_dir"])

