########################################################################################
##
##                                  TESTS FOR
##                  carrier ports of the BABY-1L measurement chain
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
from typing import ClassVar

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Amplifier
from pathsim.solvers import RKCK54

from pathsim_fusion.tritium import Bubbler4, LSCCounter, SaltCrucible
from pathsim_fusion.tritium import carrier_channels as chn
from pathsim_fusion.tritium import carrier_form as frm

from .test_lsc import DECAY_CONSTANT
from .test_salt_crucible import (
    RUN1_IRRADIATIONS,
    RUN1_K_TOP,
    RUN1_K_WALL,
    RUN1_NEUTRON_RATE,
    RUN1_RADIUS,
    RUN1_TBR,
    RUN1_TIMES_TOP,
    RUN1_VOLUME,
    exact_release,
)

# HELPERS ==============================================================================


def make_crucible(cls):
    return cls(
        radius=RUN1_RADIUS,
        height=RUN1_VOLUME / (np.pi * RUN1_RADIUS**2),
        TBR=RUN1_TBR,
        neutron_rate=RUN1_NEUTRON_RATE,
        k_top=RUN1_K_TOP,
        k_wall=RUN1_K_WALL,
        irradiations=RUN1_IRRADIATIONS,
    )


def run_chain(variant, times, fraction_soluble=0.0, ve=1.0, ce=1.0, **counter_kw):
    """Run the chain salt crucible -> bubbler -> counter with scalar ports
    ('scalar') or carrier ports ('form' or 'channels'). Returns the crucible,
    the bubbler and the counter after the last counting.

    The fraction ``fraction_soluble`` of the tritium that leaves the top
    surface is soluble. The counter is listed before the bubbler, so that it
    reads the vials before the bubbler replaces them at the same time.
    """
    times = list(times)
    kw = {
        "conversion_efficiency": ce,
        "vial_efficiency": ve,
        "replacement_times": times,
    }

    if variant == "scalar":
        crucible = make_crucible(SaltCrucible)
        bubbler = Bubbler4(**kw)
        counter = LSCCounter(count_times=times, **counter_kw)
        to_activity = Amplifier(DECAY_CONSTANT)
        soluble = Amplifier(fraction_soluble)
        insoluble = Amplifier(1.0 - fraction_soluble)
        blocks = [crucible, to_activity, soluble, insoluble, counter, bubbler]
        conns = [
            Connection(crucible["flux_top"], to_activity),
            Connection(to_activity, soluble, insoluble),
            Connection(soluble, bubbler["sample_in_soluble"]),
            Connection(insoluble, bubbler["sample_in_insoluble"]),
            *[Connection(bubbler[i], counter[i]) for i in range(4)],
        ]

    else:
        m = frm if variant == "form" else chn
        crucible = make_crucible(
            frm.SaltCrucibleForm if variant == "form" else chn.SaltCrucibleChannels
        )
        bubbler = (frm.Bubbler4Form if variant == "form" else chn.Bubbler4Channels)(
            **kw
        )
        counter = (frm.LSCCounterForm if variant == "form" else chn.LSCCounterChannels)(
            count_times=times, **counter_kw
        )
        if variant == "form":
            speciation = m.form_split(fraction_soluble)
            gain = m.stream_gain(m.Form(), DECAY_CONSTANT)
        else:
            speciation = m.form_split_channels(fraction_soluble)
            gain = m.stream_gain(m.SpeciesForm(), DECAY_CONSTANT)
        blocks = [crucible, speciation, gain, counter, bubbler]
        conns = [
            Connection(crucible["flux_top"], speciation["in"]),
            Connection(speciation["out"], gain["in"]),
            Connection(gain["out"], bubbler["sample_in"]),
            Connection(bubbler["vials"], counter["vials"]),
        ]

    sim = Simulation(
        blocks,
        conns,
        Solver=RKCK54,
        tolerance_lte_rel=1e-9,
        tolerance_lte_abs=1e-12,
        dt=100.0,
        log=False,
    )
    sim.run(times[-1] + 1000.0)
    return crucible, bubbler, counter


# TESTS ================================================================================


class TestCarrierPorts(unittest.TestCase):
    """The carrier port versions of the blocks give the results of the scalar
    port versions."""

    times: ClassVar[list[float]] = list(RUN1_TIMES_TOP)

    def test_layout(self):
        c = frm.SaltCrucibleForm.output_port_labels
        self.assertEqual(c["flux_top"].T, 4)
        self.assertEqual(c["released_top"], 12)
        b = frm.Bubbler4Form.output_port_labels
        self.assertEqual(b["vials"].vial3, 2)
        self.assertEqual(b["sample_out"].insoluble, 5)
        s = chn.Bubbler4Channels.input_port_labels["sample_in"]
        self.assertEqual((s.T_soluble, s.T_insoluble), (5, 6))

    def test_all_insoluble(self):
        """Counted activity of the chain with the form assigned downstream"""

        _, _, ref = run_chain("scalar", self.times)
        for variant in ("form", "channels"):
            with self.subTest(variant=variant):
                _, _, counter = run_chain(variant, self.times)
                np.testing.assert_array_equal(counter.net, ref.net)

    def test_mixed_forms(self):
        """Soluble and insoluble tritium with incomplete collection"""

        kw = {"fraction_soluble": 0.3, "ve": 0.9, "ce": 0.8, "background": 0.3}
        _, _, ref = run_chain("scalar", self.times, **kw)
        for variant in ("form", "channels"):
            with self.subTest(variant=variant):
                _, _, counter = run_chain(variant, self.times, **kw)
                np.testing.assert_allclose(counter.net, ref.net, rtol=1e-12)
                np.testing.assert_allclose(
                    counter.outputs["counts"], ref.outputs[:4], rtol=1e-12
                )
                # soluble and insoluble samples are counted separately
                self.assertGreater(counter.outputs["counts"][0], 0.0)

    def test_sample_out(self):
        """The sample leaving the bubbler is the sum over the forms"""

        kw = {"fraction_soluble": 0.3, "ve": 0.9, "ce": 0.8}
        _, ref, _ = run_chain("scalar", self.times, **kw)
        _, form, _ = run_chain("form", self.times, **kw)
        _, chan, _ = run_chain("channels", self.times, **kw)
        out = ref.outputs[4]
        np.testing.assert_allclose(form.outputs["sample_out"].sum(), out, rtol=1e-12)
        np.testing.assert_allclose(chan.outputs["sample_out"][2], out, rtol=1e-12)

    def test_release_closed_form(self):
        """The counted activity follows the closed form of the model"""

        crucible, _, counter = run_chain("form", self.times)
        counted = np.cumsum([n.sum() for n in counter.net])
        exact = DECAY_CONSTANT * exact_release(crucible, self.times)[1]
        np.testing.assert_allclose(counted, exact, rtol=1e-6)

    def test_crucible_outputs(self):
        """Released and flux channels of the crucible are those of the scalar version"""

        ref, _, _ = run_chain("scalar", self.times)
        for variant, key in (("form", "T"), ("channels", "T")):
            with self.subTest(variant=variant):
                cr, _, _ = run_chain(variant, self.times)
                self.assertEqual(
                    cr.outputs["released_top"], ref.outputs["released_top"]
                )
                self.assertEqual(
                    cr.outputs["released_wall"], ref.outputs["released_wall"]
                )
                self.assertEqual(cr.outputs["flux_top"][2], ref.outputs["flux_top"])
                np.testing.assert_array_equal(cr.outputs["flux_top"][[0, 1, 3, 4]], 0.0)

    def test_connection_widths(self):
        """A stream does not connect to a port of another width"""

        cr = make_crucible(frm.SaltCrucibleForm)
        bub = frm.Bubbler4Form()
        with self.assertRaises(ValueError):
            Connection(cr["flux_top"], bub["sample_in"])


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
