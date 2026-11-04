"""Variant (b): the chemical form is extra channels on the one carrier,
``SpeciesForm``."""

from typing import ClassVar

from pathsim import Carrier

from ._carrier_blocks import CarrierBubbler4, CarrierLSCCounter, CarrierSaltCrucible
from .streams import (
    Counts,
    SpeciesForm,
    Vials,
    form_split_channels,
    stream_gain,
)

_n = len(SpeciesForm())
_flux_top = SpeciesForm(2)
_flux_wall = SpeciesForm(2 + _n)
_rel = _flux_wall.stop


class SaltCrucibleChannels(CarrierSaltCrucible):
    """`SaltCrucible` with ``flux_top`` and ``flux_wall`` as ``SpeciesForm``,
    only ``T`` is written, the form is assigned downstream."""

    output_port_labels: ClassVar[dict[str, Carrier | int]] = {
        "concentration": 0,
        "inventory": 1,
        "flux_top": _flux_top,
        "flux_wall": _flux_wall,
        "released_top": _rel,
        "released_wall": _rel + 1,
    }


class Bubbler4Channels(CarrierBubbler4):
    """`Bubbler4` with the ``SpeciesForm`` stream ``sample_in`` as input."""

    _in = SpeciesForm(0)
    _out = SpeciesForm(4)
    _sol = "T_soluble"
    _insol = "T_insoluble"
    _total = "T"
    input_port_labels: ClassVar[dict[str, Carrier | int]] = {"sample_in": _in}
    output_port_labels: ClassVar[dict[str, Carrier | int]] = {
        "vials": Vials(0),
        "sample_out": _out,
    }


LSCCounterChannels = CarrierLSCCounter

__all__ = [
    "Bubbler4Channels",
    "Counts",
    "LSCCounterChannels",
    "SaltCrucibleChannels",
    "SpeciesForm",
    "Vials",
    "form_split_channels",
    "stream_gain",
]
