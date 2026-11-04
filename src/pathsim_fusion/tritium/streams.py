#########################################################################################
##
##                       Stream carriers of the tritium blocks
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
from pathsim import Carrier
from pathsim.blocks._block import Block
from pathsim.optim.operator import Operator

# CARRIERS ==============================================================================


class Species(Carrier):
    """Isotope stream, in particles/s (atoms, so T2 counts twice)."""

    keys = ("H", "D", "T", "He", "Z")


class Form(Carrier):
    """Tritium stream split by chemical form, variant (a): second carrier."""

    keys = ("soluble", "insoluble")


class SpeciesForm(Carrier):
    """Isotope stream with the tritium split by form, variant (b): extra
    channels on the one carrier. ``T`` is the total, ``T_soluble`` and
    ``T_insoluble`` the parts of it by form."""

    keys = ("H", "D", "T", "He", "Z", "T_soluble", "T_insoluble")


class Vials(Carrier):
    """Content of the four bubbler vials."""

    keys = ("vial1", "vial2", "vial3", "vial4")


class Counts(Carrier):
    """Net activity read by the scintillation counter."""

    keys = ("soluble", "insoluble", "total", "cumulative")


# BLOCKS ================================================================================


class StreamMap(Block):
    """Linear map between two streams, ``out = matrix @ in``.

    Used for the unit conversion (particles/s to Bq) and for assigning the
    chemical form to a stream.

    Parameters
    ----------
    source : Carrier
        carrier of the input port
    target : Carrier
        carrier of the output port
    matrix : array_like
        matrix of shape ``(len(target), len(source))``
    """

    def __init__(self, source, target, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (len(target), len(source)):
            raise ValueError(
                f"'matrix' must have shape {(len(target), len(source))} "
                f"but has {matrix.shape}"
            )
        self.input_port_labels = {"in": Carrier(0, keys=source.keys)}
        self.output_port_labels = {"out": Carrier(0, keys=target.keys)}
        super().__init__()
        self.matrix = matrix
        self.op_alg = Operator(
            func=lambda u: self.matrix @ u, jac=lambda u: self.matrix
        )

    def update(self, t):
        self.outputs.update_from_array(self.op_alg(self.inputs.to_array()))


def form_split(fraction_soluble):
    """Assign the form to a ``Species`` stream, variant (a).

    The tritium of the stream is split into a ``Form`` stream, a fraction
    ``fraction_soluble`` is soluble.
    """
    m = np.zeros((len(Form()), len(Species())))
    m[0, Species().T] = fraction_soluble
    m[1, Species().T] = 1.0 - fraction_soluble
    return StreamMap(Species(), Form(), m)


def form_split_channels(fraction_soluble):
    """Assign the form to the tritium of a ``SpeciesForm`` stream, variant (b).

    The other isotopes pass, ``T`` is kept and split into ``T_soluble`` and
    ``T_insoluble``.
    """
    s = SpeciesForm()
    m = np.eye(len(s))
    m[s.T_soluble, :] = 0.0
    m[s.T_insoluble, :] = 0.0
    m[s.T_soluble, s.T] = fraction_soluble
    m[s.T_insoluble, s.T] = 1.0 - fraction_soluble
    return StreamMap(SpeciesForm(), SpeciesForm(), m)


def stream_gain(carrier, gain):
    """Scale all channels of a stream, for example from particles/s to Bq."""
    return StreamMap(carrier, carrier, gain * np.eye(len(carrier)))
