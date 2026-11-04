#########################################################################################
##
##                          Liquid Scintillation Counter Block
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

import numpy as np
from pathsim.blocks._block import Block
from pathsim.events.schedule import ScheduleList

# BLOCK DEFINITIONS =====================================================================


class LSCCounter(Block):
    """Liquid scintillation counting of the four vials of a bubbler sample.

    Discrete event block that models the measurement chain after a
    :class:`~pathsim_fusion.tritium.bubbler.Bubbler4`. At every counting time
    the activity of the four vials is read, the counter adds its background to
    get the gross reading, the analyst subtracts a background and the net
    values are summed and held until the next count.

    Mathematical Formulation
    -------------------------
    For vial :math:`j` at counting :math:`i` with input activity
    :math:`a_{ij}` and quench indicator :math:`q_{ij}` (tSIE)

    .. math::

        g_{ij} = a_{ij} + b(q_{ij}) \\qquad
        n_{ij} = \\max(g_{ij} - b_s(q_{ij}), 0)

    where :math:`b` is the ``background`` of the counter and :math:`b_s` the
    ``subtracted_background`` of the analysis. The two are the same unless the
    analysis subtracts a different background than the counter has, for
    example one blank vial for all vials while the real background depends on
    the quench of each vial. The soluble and insoluble samples and the total are

    .. math::

        n_{sol} = n_{i1} + n_{i2}, \\quad
        n_{insol} = n_{i3} + n_{i4}, \\quad
        n_{tot} = n_{sol} + n_{insol}

    and the output ``cumulative`` is the sum of all totals counted so far.

    Note
    ----
    The input should be the vial inventory of the bubbler as an activity
    (decay constant times the number of tritium atoms). The vials are read at
    the counting times, which are normally the vial replacement times of the
    bubbler. Add the counter to the simulation before the bubbler, so that the
    vials are read before the bubbler resets them at the same time.

    The clipping of negative net values at zero follows ``libra_toolbox``.

    Parameters
    ----------
    count_times : list[float]
        times at which the vials are counted, in ascending order
    background : float | callable
        background of the counter in the units of the input. A constant or a
        function ``b(tsie)`` of the quench indicator. Default 0.
    subtracted_background : float | callable | None
        background subtracted in the analysis, same form as ``background``.
        Default None (same as ``background``).
    tsie : float | list | None
        quench indicator of the vials. A single value for all vials, one value
        per counting or one value per vial and counting (shape
        ``(len(count_times), 4)``). Needed if a background is a function.
    clip_negative : bool
        set negative net vial values to zero. Default True.

    Attributes
    ----------
    gross : list[numpy.ndarray]
        gross reading of the four vials at each counting so far
    net : list[numpy.ndarray]
        net reading of the four vials at each counting so far

    Inputs
    ------
    vial1 (0) : activity in vial 1
    vial2 (1) : activity in vial 2
    vial3 (2) : activity in vial 3
    vial4 (3) : activity in vial 4

    Outputs
    -------
    soluble (0) : net activity of vials 1 and 2 at the last counting
    insoluble (1) : net activity of vials 3 and 4 at the last counting
    total (2) : net activity of all vials at the last counting
    cumulative (3) : sum of the total net activity of all countings so far
    """

    # discrete time block -> no continuous time linear model
    linearizable = False

    input_port_labels: ClassVar[dict[str, int]] = {
        "vial1": 0,
        "vial2": 1,
        "vial3": 2,
        "vial4": 3,
    }
    output_port_labels: ClassVar[dict[str, int]] = {
        "soluble": 0,
        "insoluble": 1,
        "total": 2,
        "cumulative": 3,
    }

    def __init__(
        self,
        count_times,
        background=0.0,
        subtracted_background=None,
        tsie=None,
        clip_negative=True,
    ):
        super().__init__()

        count_times = np.atleast_1d(np.asarray(count_times, dtype=float))
        if count_times.size == 0:
            raise ValueError("'count_times' must not be empty")

        self.count_times = count_times.tolist()
        self.background = background
        self.subtracted_background = (
            background if subtracted_background is None else subtracted_background
        )
        self.clip_negative = clip_negative

        # quench indicator for every vial at every counting
        if tsie is None:
            if callable(self.background) or callable(self.subtracted_background):
                raise ValueError("'tsie' is required if a background is a function")
            self.tsie = None
        else:
            tsie = np.asarray(tsie, dtype=float)
            try:
                tsie = np.broadcast_to(
                    tsie.reshape(-1, 1) if tsie.ndim == 1 else tsie,
                    (count_times.size, 4),
                )
            except ValueError:
                raise ValueError(
                    "'tsie' must be a single value, one value per counting or "
                    f"shape ({count_times.size}, 4) but has shape {tsie.shape}"
                ) from None
            self.tsie = tsie.copy()

        self.gross = []
        self.net = []
        self._cumulative = 0.0

        self.events = [ScheduleList(times_evt=self.count_times, func_act=self._count)]

    def __len__(self):
        # the outputs only change at events, no algebraic path
        return 0

    def _background_at(self, background, i):
        """Background for the four vials at the i-th counting."""
        if callable(background):
            return np.array([background(q) for q in self.tsie[i]], dtype=float)
        return np.full(4, float(background))

    def _count(self, t):
        """Count the four vials and update the held outputs."""

        i = len(self.net)
        activity = self.inputs.to_array()[:4]

        gross = activity + self._background_at(self.background, i)
        net = gross - self._background_at(self.subtracted_background, i)
        if self.clip_negative:
            net = np.clip(net, 0.0, None)

        self.gross.append(gross)
        self.net.append(net)

        soluble = net[0] + net[1]
        insoluble = net[2] + net[3]
        self._cumulative += soluble + insoluble

        self.outputs.update_from_array(
            np.array([soluble, insoluble, soluble + insoluble, self._cumulative])
        )

    def reset(self):
        """Reset the block and forget all countings."""
        super().reset()
        self.gross = []
        self.net = []
        self._cumulative = 0.0
