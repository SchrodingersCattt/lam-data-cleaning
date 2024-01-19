from dflow.python import OP, OPIO, OPIOSign, Parameter


class Merge(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "loop_lcurve": Parameter(dict, default={}),
            "zero_lcurve": Parameter(dict, default={}),
            "all_lcurve": Parameter(dict, default={}),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "lcurve": dict,
        })

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        lcurve = {}
        for lc in [ip["zero_lcurve"], ip["loop_lcurve"], ip["all_lcurve"]]:
            for k, v in lc.items():
                lcurve[k] = lcurve.get(k, []) + v

        return OPIO({
            "lcurve": lcurve,
        })
