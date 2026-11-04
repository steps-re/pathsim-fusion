"""Variant (a): the chemical form is a second carrier, ``Form``."""

from typing import ClassVar

from pathsim import Carrier

from ._carrier_blocks import CarrierBubbler4, CarrierLSCCounter, CarrierSaltCrucible
from .streams import Counts, Form, Species, Vials, form_split, stream_gain

_flux_top = Species(2)
_flux_wall = Species(2 + len(Species()))
_rel = _flux_wall.stop


class SaltCrucibleForm(CarrierSaltCrucible):
    """`SaltCrucible` with ``flux_top`` and ``flux_wall`` as ``Species``."""

    output_port_labels: ClassVar[dict[str, Carrier | int]] = {
        "concentration": 0,
        "inventory": 1,
        "flux_top": _flux_top,
        "flux_wall": _flux_wall,
        "released_top": _rel,
        "released_wall": _rel + 1,
    }


class Bubbler4Form(CarrierBubbler4):
    """`Bubbler4` with the ``Form`` stream ``sample_in`` as input."""

    _in = Form(0)
    _out = Form(4)
    _sol = "soluble"
    _insol = "insoluble"
    input_port_labels: ClassVar[dict[str, Carrier | int]] = {"sample_in": _in}
    output_port_labels: ClassVar[dict[str, Carrier | int]] = {
        "vials": Vials(0),
        "sample_out": _out,
    }


LSCCounterForm = CarrierLSCCounter

__all__ = [
    "Bubbler4Form",
    "Counts",
    "Form",
    "LSCCounterForm",
    "SaltCrucibleForm",
    "Species",
    "Vials",
    "form_split",
    "stream_gain",
]
