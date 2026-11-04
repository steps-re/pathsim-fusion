#########################################################################################
##
##              Carrier port versions of the BABY-1L measurement chain
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

import numpy as np
from pathsim import Carrier
from pathsim.optim.operator import DynamicOperator

from .bubbler import Bubbler4
from .lsc import LSCCounter
from .salt_crucible import SaltCrucible
from .streams import Counts, Vials

# BLOCKS ================================================================================


class CarrierSaltCrucible(SaltCrucible):
    """`SaltCrucible` with the release rates as streams of a carrier.

    The release is tritium only, so only the ``T`` channel of ``flux_top`` and
    ``flux_wall`` is written, the other channels stay zero. Subclasses set the
    port labels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        top = self.output_port_labels["flux_top"]
        wall = self.output_port_labels["flux_wall"]
        i_top, i_wall = top["T"], wall["T"]
        i_rel = max(top.stop, wall.stop)

        def _fn_a(x, u, t):
            q_top, q_wall = self._release_rates(x[0], t)
            y = np.zeros(i_rel + 2)
            y[0], y[1] = x[0], self.volume * x[0]
            y[i_top], y[i_wall] = q_top, q_wall
            y[i_rel], y[i_rel + 1] = x[1], x[2]
            return y

        self.func_alg = _fn_a
        self.op_alg = DynamicOperator(func=_fn_a)


class CarrierBubbler4(Bubbler4):
    """`Bubbler4` with one input stream and the vials and the sample as
    streams.

    The soluble and insoluble tritium are the channels ``_sol`` and ``_insol``
    of the input carrier, the sample leaving the bubbler is written to the
    same channels of the output carrier, and to ``_total`` if that is set.
    Subclasses set the carriers and port labels.
    """

    _in: Carrier
    _out: Carrier
    _sol: str
    _insol: str
    _total: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        i_sol, i_insol = self._in[self._sol], self._in[self._insol]
        o_sol, o_insol = self._out[self._sol], self._out[self._insol]
        o_total = None if self._total is None else self._out[self._total]
        n_out = self._out.stop

        def _fn_d(x, u, t):
            ve, ce = self.vial_efficiency, self.conversion_efficiency
            sol, ins = u[i_sol], u[i_insol]
            dv1 = ve * sol
            dv2 = dv1 * (1 - ve)
            dv3 = ve * (ce * ins + (1 - ve) ** 2 * sol)
            dv4 = dv3 * (1 - ve)
            return np.array([dv1, dv2, dv3, dv4])

        def _fn_a(x, u, t):
            ve, ce = self.vial_efficiency, self.conversion_efficiency
            sol, ins = u[i_sol], u[i_insol]
            y = np.zeros(n_out)
            y[:4] = x
            y[o_sol] = (1 - ve) ** 2 * (ce * ins + (1 - ve) ** 2 * sol)
            y[o_insol] = (1 - ce) * ins
            if o_total is not None:
                y[o_total] = y[o_sol] + y[o_insol]
            return y

        self.func_dyn = _fn_d
        self.func_alg = _fn_a
        self.op_dyn = DynamicOperator(func=_fn_d, jac_x=self.jac_dyn)
        self.op_alg = DynamicOperator(func=_fn_a)


class CarrierLSCCounter(LSCCounter):
    """`LSCCounter` reading the vials as one stream and writing the counts as
    one carrier. The channel layout is the one of the scalar version.
    """

    input_port_labels: ClassVar[dict[str, Carrier | int]] = {"vials": Vials(0)}
    output_port_labels: ClassVar[dict[str, Carrier | int]] = {"counts": Counts(0)}
