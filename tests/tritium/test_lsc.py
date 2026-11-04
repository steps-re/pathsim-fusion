########################################################################################
##
##                                  TESTS FOR
##                              'tritium.lsc.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Amplifier, Constant
from pathsim.solvers import EUF, RKCK54

from pathsim_fusion.tritium import Bubbler4, LSCCounter

from .test_salt_crucible import (
    LIBRA_RELEASE_TOP,
    RUN1_TIMES_TOP,
    exact_release,
    run1_crucible,
)

# HELPERS ==============================================================================

# Decay constant of tritium in 1/s as used by the libra toolbox to convert a number of
# particles to an activity (3.57e14 Bq/g, 3.016 g/mol)
DECAY_CONSTANT = 3.57e14 * (6.032 / 2) / 6.02214076e23


def run_counter(counter, activities, duration, dt=0.1):
    """Feed constant vial activities into the counter and run."""
    sources = [Constant(a) for a in activities]
    sim = Simulation(
        [*sources, counter],
        [Connection(s, counter[i]) for i, s in enumerate(sources)],
        Solver=EUF,
        dt=dt,
        log=False,
    )
    sim.run(duration)
    return sim


# TESTS ================================================================================


class TestTritiumLSCCounter(unittest.TestCase):
    """
    Test the implementation of the 'LSCCounter' block class from the fusion toolbox.
    The block is a discrete event block that counts the four vials of a bubbler sample.
    """

    def test_init(self):
        """Test initialization with various parameter combinations"""

        C = LSCCounter(count_times=[1.0, 2.0])
        self.assertEqual(C.count_times, [1.0, 2.0])
        self.assertEqual(C.background, 0.0)
        self.assertEqual(C.subtracted_background, 0.0)
        self.assertIsNone(C.tsie)
        self.assertTrue(C.clip_negative)
        self.assertEqual(len(C.events), 1)
        self.assertEqual(len(C), 0)

        # the analysis subtracts the counter background by default
        C = LSCCounter(count_times=5.0, background=0.3)
        self.assertEqual(C.count_times, [5.0])
        self.assertEqual(C.subtracted_background, 0.3)

        # quench indicator broadcast to every counting and vial
        C = LSCCounter(count_times=[1.0, 2.0], tsie=300.0)
        self.assertEqual(C.tsie.shape, (2, 4))
        C = LSCCounter(count_times=[1.0, 2.0], tsie=[300.0, 400.0])
        np.testing.assert_array_equal(C.tsie[:, 3], [300.0, 400.0])
        C = LSCCounter(count_times=[1.0, 2.0], tsie=np.ones((2, 4)))
        self.assertEqual(C.tsie.shape, (2, 4))

        # initial outputs
        np.testing.assert_array_equal(C.outputs.to_array(), np.zeros(4))

    def test_init_validation(self):
        """Test that invalid parameters are rejected"""

        with self.assertRaises(ValueError):
            LSCCounter(count_times=[])

        # a background that depends on the quench needs the quench
        with self.assertRaises(ValueError):
            LSCCounter(count_times=[1.0], background=lambda q: 0.3)
        with self.assertRaises(ValueError):
            LSCCounter(count_times=[1.0], subtracted_background=lambda q: 0.3)

        # wrong shape of the quench
        with self.assertRaises(ValueError):
            LSCCounter(count_times=[1.0, 2.0], tsie=[300.0, 300.0, 300.0])
        with self.assertRaises(ValueError):
            LSCCounter(count_times=[1.0, 2.0], tsie=np.ones((2, 3)))

        # times in descending order are caught by the event
        with self.assertRaises(ValueError):
            LSCCounter(count_times=[2.0, 1.0])

    def test_counting(self):
        """The outputs hold the counts of the vials until the next count"""

        C = LSCCounter(count_times=[1.0, 2.0])
        sim = run_counter(C, [1.0, 2.0, 4.0, 8.0], 0.5)

        # nothing counted yet
        np.testing.assert_array_equal(C.outputs.to_array(), np.zeros(4))
        self.assertEqual(len(C.net), 0)

        sim.run(1.0)
        np.testing.assert_allclose(C.outputs.to_array(), [3.0, 12.0, 15.0, 15.0])
        np.testing.assert_allclose(C.net[0], [1.0, 2.0, 4.0, 8.0])

        # held between the counts, the cumulative output adds the second counting
        sim.run(0.25)
        np.testing.assert_allclose(C.outputs.to_array(), [3.0, 12.0, 15.0, 15.0])
        sim.run(1.0)
        np.testing.assert_allclose(C.outputs.to_array(), [3.0, 12.0, 15.0, 30.0])
        self.assertEqual(len(C.net), 2)

        # port labels
        self.assertEqual(LSCCounter.input_port_labels["vial4"], 3)
        self.assertEqual(LSCCounter.output_port_labels["cumulative"], 3)

    def test_same_background(self):
        """A background that is added and subtracted again leaves the activity"""

        C = LSCCounter(count_times=[1.0], background=0.3)
        run_counter(C, [1.0, 2.0, 4.0, 8.0], 1.5)

        np.testing.assert_allclose(C.gross[0], [1.3, 2.3, 4.3, 8.3])
        np.testing.assert_allclose(C.net[0], [1.0, 2.0, 4.0, 8.0])

    def test_wrong_background(self):
        """A different subtracted background biases every vial"""

        C = LSCCounter(count_times=[1.0], background=0.3, subtracted_background=0.4)
        run_counter(C, [1.0, 2.0, 4.0, 8.0], 1.5)

        np.testing.assert_allclose(C.net[0], [0.9, 1.9, 3.9, 7.9])
        np.testing.assert_allclose(C.outputs["total"], 14.6)

    def test_negative_clipping(self):
        """Negative net values are set to zero, as in the libra toolbox"""

        C = LSCCounter(count_times=[1.0], background=0.1, subtracted_background=0.5)
        run_counter(C, [0.1, 1.0, 0.0, 2.0], 1.5)
        np.testing.assert_allclose(C.net[0], [0.0, 0.6, 0.0, 1.6])
        np.testing.assert_allclose(C.outputs.to_array()[:3], [0.6, 1.6, 2.2])

        C = LSCCounter(
            count_times=[1.0],
            background=0.1,
            subtracted_background=0.5,
            clip_negative=False,
        )
        run_counter(C, [0.1, 1.0, 0.0, 2.0], 1.5)
        np.testing.assert_allclose(C.net[0], [-0.3, 0.6, -0.4, 1.6])

    def test_quench_dependent_background(self):
        """The background follows the quench of each vial, the analysis does not

        The curve is the interpolation of a blank scan, a lower background
        for a more quenched vial (higher tSIE), while one blank vial counted at
        tSIE 290 is subtracted from all vials.
        """

        scan_tsie = [278.0, 324.0, 489.0]
        scan_bq = [0.31, 0.24, 0.17]

        def curve(tsie):
            return np.interp(tsie, scan_tsie, scan_bq)

        # vial 1 of the second counting is the evaporated one, high tSIE
        tsie = [[290.0] * 4, [489.0, 290.0, 290.0, 290.0]]
        blank = float(curve(290.0))

        C = LSCCounter(
            count_times=[1.0, 2.0],
            background=curve,
            subtracted_background=blank,
            tsie=tsie,
        )
        run_counter(C, [1.0, 1.0, 1.0, 1.0], 2.5)

        # unquenched vials are unbiased
        np.testing.assert_allclose(C.net[0], np.ones(4))
        np.testing.assert_allclose(C.net[1][1:], np.ones(3))

        # the high tSIE vial is over-subtracted by the blank
        np.testing.assert_allclose(C.net[1][0], 1.0 + 0.17 - blank)
        self.assertLess(C.net[1][0], 1.0)

        # the matching curve is unbiased
        C = LSCCounter(count_times=[1.0, 2.0], background=curve, tsie=tsie)
        run_counter(C, [1.0, 1.0, 1.0, 1.0], 2.5)
        np.testing.assert_allclose(C.net[1], np.ones(4))

    def test_reset(self):
        """Reset forgets all countings"""

        C = LSCCounter(count_times=[1.0, 2.0])
        sim = run_counter(C, [1.0, 2.0, 4.0, 8.0], 2.5)
        self.assertEqual(len(C.net), 2)
        self.assertGreater(C.outputs["cumulative"], 0.0)

        sim.reset()
        self.assertEqual(C.net, [])
        self.assertEqual(C.gross, [])
        np.testing.assert_array_equal(C.outputs.to_array(), np.zeros(4))

        # same result when run again
        sim.run(2.5)
        self.assertEqual(len(C.net), 2)
        np.testing.assert_allclose(C.outputs["cumulative"], 30.0)


class TestBabyMeasurementChain(unittest.TestCase):
    """
    Test the chain salt crucible -> bubbler -> liquid scintillation counter, that
    models the tritium measurements of BABY-1L run 1 from the release of the salt
    to the activity counted in the vials of the top (IV) gas stream.
    """

    def run_chain(
        self, times, vial_efficiency=1.0, conversion_efficiency=1.0, **kwargs
    ):
        """All tritium leaving the top surface is insoluble and goes through the
        bubbler. The counter is listed before the bubbler, so that it reads the
        vials before the bubbler replaces them at the same time.
        """

        crucible = run1_crucible()
        to_activity = Amplifier(DECAY_CONSTANT)
        no_soluble = Constant(0.0)
        bubbler = Bubbler4(
            conversion_efficiency=conversion_efficiency,
            vial_efficiency=vial_efficiency,
            replacement_times=list(times),
        )
        counter = LSCCounter(count_times=times, **kwargs)

        sim = Simulation(
            [crucible, to_activity, no_soluble, counter, bubbler],
            [
                Connection(crucible["flux_top"], to_activity),
                Connection(to_activity, bubbler["sample_in_insoluble"]),
                Connection(no_soluble, bubbler["sample_in_soluble"]),
                *[Connection(bubbler[i], counter[i]) for i in range(4)],
            ],
            Solver=RKCK54,
            tolerance_lte_rel=1e-9,
            tolerance_lte_abs=1e-12,
            dt=100.0,
            log=False,
        )
        sim.run(times[-1] + 1000.0)

        return crucible, counter

    def test_cumulative_release(self):
        """The counted activity adds up to the cumulative release of the salt"""

        times = list(RUN1_TIMES_TOP)
        crucible, counter = self.run_chain(times)

        counted = np.cumsum([n.sum() for n in counter.net])
        exact = DECAY_CONSTANT * exact_release(crucible, times)[1]

        # the closed form of the model
        np.testing.assert_allclose(counted, exact, rtol=1e-6)

        # the libra toolbox, within its solver error of 1%
        np.testing.assert_allclose(
            counted, DECAY_CONSTANT * LIBRA_RELEASE_TOP, rtol=1e-2
        )

        # everything is insoluble
        np.testing.assert_array_equal([n[:2].sum() for n in counter.net], 0.0)
        np.testing.assert_allclose(counter.outputs["cumulative"], counted[-1])

        # final cumulative activity is about 16 Bq, as measured
        self.assertAlmostEqual(counted[-1], 16.0, delta=0.5)

    def test_incomplete_collection(self):
        """Vial and conversion efficiency scale the counted activity"""

        times = list(RUN1_TIMES_TOP)
        ve, ce = 0.9, 0.8
        _, full = self.run_chain(times)
        _, partial = self.run_chain(times, vial_efficiency=ve, conversion_efficiency=ce)

        # vials 3 and 4 in series collect a fraction ve (2 - ve) of the converted flux
        counted_full = np.cumsum([n.sum() for n in full.net])
        counted_partial = np.cumsum([n.sum() for n in partial.net])
        np.testing.assert_allclose(
            counted_partial, ce * ve * (2 - ve) * counted_full, rtol=1e-6
        )

    def test_background_bias(self):
        """Subtracting a larger background than the counter has lowers the total"""

        times = list(RUN1_TIMES_TOP)
        _, ideal = self.run_chain(times, background=0.3)
        _, biased = self.run_chain(times, background=0.3, subtracted_background=0.32)

        # with full efficiency all tritium is counted in the third vial, so each
        # counting loses the difference of the backgrounds once
        ideal_total = sum(n.sum() for n in ideal.net)
        biased_total = sum(n.sum() for n in biased.net)

        self.assertAlmostEqual(
            ideal_total - biased_total, 0.02 * len(times), delta=1e-6
        )


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
