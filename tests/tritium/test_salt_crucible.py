########################################################################################
##
##                                  TESTS FOR
##                           'tritium.salt_crucible.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
from pathsim import Connection, Simulation
from pathsim.blocks import Scope
from pathsim.solvers import ESDIRK43, RKBS32, RKCK54, SSPRK22
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pathsim_fusion.tritium import SaltCrucible

# HELPERS ==============================================================================


def exact_release(crucible, t_eval):
    """Closed form of the model for constant coefficients, evaluated at the
    times `t_eval`. The state is propagated through every irradiation and
    decay interval with a matrix exponential. Returns the arrays of
    concentration, top release and wall release.
    """

    V = crucible.volume
    a_top = crucible.A_top * crucible.k_top
    a_wall = crucible.A_wall * crucible.k_wall
    S = crucible.TBR * crucible.neutron_rate

    # state is [c, N_top, N_wall, 1], the constant source enters via the last entry
    def generator(on):
        return np.array(
            [
                [-(a_top + a_wall) / V, 0, 0, on * S / V],
                [a_top, 0, 0, 0],
                [a_wall, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

    edges = sorted(
        {0.0} | {t for window in crucible.irradiations for t in window if t < np.inf}
    )

    def state_at(t):
        x = np.array([crucible.initial_concentration, 0.0, 0.0, 1.0])
        for t0, t1 in zip(edges, edges[1:] + [np.inf]):
            if t <= t0:
                break
            on = any(start <= t0 < stop for start, stop in crucible.irradiations)
            x = expm(generator(on) * (min(t, t1) - t0)) @ x
        return x[:3]

    return np.array([state_at(t) for t in t_eval]).T


def run_to(sim, times):
    """Advance the simulation to every time in `times` and yield after each."""
    for t in times:
        sim.run(t - sim.time)
        # a run shorter than the current timestep would overshoot silently
        assert abs(sim.time - t) < 1e-6, f"simulation at {sim.time}, expected {t}"
        yield t


# BABY-1L RUN 1 ========================================================================

# Inputs of the tritium model of BABY-1L run 1 as processed by the lab
# (LIBRA-project/BABY-1L-run-1, data/processed_data.json, after the salt mass
# fix: 1.88 kg of salt give 1.22 L, k_top 1.09e-7 m/s, wall ratio 1.49e-2)
RUN1_RADIUS = 0.07
RUN1_VOLUME = 1.2209397030244564e-3
RUN1_TBR = 0.0025476461715254082
RUN1_NEUTRON_RATE = 85833760.6445419
RUN1_K_TOP = 1.09e-7
RUN1_K_WALL = 1.49e-2 * RUN1_K_TOP
RUN1_IRRADIATIONS = [(0.0, 35580.0), (36000.0, 43620.0)]

# Sampling times of the top (IV) and wall (OV) gas streams in seconds
RUN1_TIMES_TOP = np.array(
    [95100, 254520, 530760, 793440, 1222380, 1663920, 2165760, 2780100, 3031560.0]
)
RUN1_TIMES_WALL = RUN1_TIMES_TOP[3:]

# Cumulative release in particles at the sampling times, from
# `libra_toolbox.tritium.model.Model` v0.10 with its default solver settings
# (BDF, rtol 1e-3) run to 50 days, linearly interpolated onto the sampling times
LIBRA_RELEASE_TOP = np.array(
    [
        8.983853e8,
        2.550238e9,
        4.647264e9,
        5.998052e9,
        7.352316e9,
        8.097558e9,
        8.528599e9,
        8.767052e9,
        8.817956e9,
    ]
)
LIBRA_RELEASE_WALL = np.array(
    [2.963043e8, 3.632051e8, 4.000201e8, 4.213135e8, 4.330932e8, 4.356078e8]
)


def run1_crucible():
    height = RUN1_VOLUME / (np.pi * RUN1_RADIUS**2)
    return SaltCrucible(
        radius=RUN1_RADIUS,
        height=height,
        TBR=RUN1_TBR,
        neutron_rate=RUN1_NEUTRON_RATE,
        k_top=RUN1_K_TOP,
        k_wall=RUN1_K_WALL,
        irradiations=RUN1_IRRADIATIONS,
    )


# TESTS ================================================================================


class TestTritiumSaltCrucible(unittest.TestCase):
    """
    Test the implementation of the 'SaltCrucible' block class from the fusion
    toolbox. The block inherits from `DynamicalSystem` and models the zero
    dimensional tritium release from a salt filled crucible.
    """

    def make(self, **kwargs):
        defaults = {
            "radius": 0.07,
            "height": 0.08,
            "TBR": 2e-3,
            "neutron_rate": 1e8,
            "k_top": 1e-5,
            "k_wall": 2e-6,
            "irradiations": [(0.0, np.inf)],
        }
        defaults.update(kwargs)
        return SaltCrucible(**defaults)

    def simulate(self, crucible, duration, Solver=RKCK54, **kwargs):
        """Run the block alone and return times and the 6 outputs."""
        sco = Scope()
        sim = Simulation(
            [crucible, sco],
            [Connection(crucible[i], sco[i]) for i in range(6)],
            Solver=Solver,
            log=False,
            **kwargs,
        )
        sim.run(duration)
        return sco.read()

    def test_init(self):
        """Test initialization and geometry"""

        B = self.make(irradiations=[(0, 10), (20, 30)])
        self.assertEqual(B.radius, 0.07)
        self.assertEqual(B.height, 0.08)
        self.assertEqual(B.irradiations, [(0, 10), (20, 30)])
        self.assertAlmostEqual(B.wall_thickness, 1.524e-3)

        # geometry as in the reference model
        self.assertAlmostEqual(B.volume, np.pi * 0.07**2 * 0.08)
        self.assertAlmostEqual(B.A_top, np.pi * 0.07**2)
        self.assertAlmostEqual(
            B.A_wall, 2 * np.pi * (0.07 + 1.524e-3) * 0.08 + np.pi * 0.07**2
        )

        # one event for all unique switching times
        self.assertEqual(len(B.events), 1)

        # no irradiation, no events
        B = self.make(irradiations=None)
        self.assertEqual(len(B.events), 0)

        # an unbounded irradiation has only the start
        B = self.make(irradiations=[(0.0, np.inf)])
        self.assertEqual(len(B.events), 0)

        # engine initial value
        B.set_solver(RKCK54, parent=None)
        np.testing.assert_array_equal(B.engine.initial_value, np.zeros(3))

        # initial concentration
        B = self.make(initial_concentration=5.0)
        B.set_solver(RKCK54, parent=None)
        np.testing.assert_array_equal(B.engine.initial_value, [5.0, 0.0, 0.0])

    def test_init_validation(self):
        """Test that invalid parameters are rejected"""

        for kwargs in [
            {"radius": 0.0},
            {"height": -1.0},
            {"TBR": -1e-3},
            {"neutron_rate": -1.0},
            {"k_top": -1e-7},
            {"k_wall": -1e-7},
            {"wall_thickness": -1.0},
            {"irradiations": [(10.0, 5.0)]},
            {"irradiations": [(10.0, 10.0)]},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_from_salt_mass(self):
        """Test the salt height from the salt mass"""

        B = SaltCrucible.from_salt_mass(
            salt_mass=1.88,
            salt_density=1539.8,
            radius=0.07,
            TBR=2e-3,
            neutron_rate=1e8,
            k_top=1e-7,
            k_wall=1e-9,
        )
        self.assertAlmostEqual(B.volume, 1.88 / 1539.8)
        self.assertAlmostEqual(B.height, 1.88 / 1539.8 / (np.pi * 0.07**2))

        with self.assertRaises(ValueError):
            SaltCrucible.from_salt_mass(
                salt_mass=0.0,
                salt_density=1539.8,
                radius=0.07,
                TBR=2e-3,
                neutron_rate=1e8,
                k_top=1e-7,
                k_wall=1e-9,
            )

    def test_update_outputs(self):
        """Test the output calculation for a given state"""

        B = self.make(initial_concentration=1e12)
        B.set_solver(RKCK54, parent=None)
        B.update(0.0)

        c = 1e12
        q_top = B.A_top * 1e-5 * c
        q_wall = B.A_wall * 2e-6 * c
        np.testing.assert_allclose(
            B.outputs.to_array(),
            [c, B.volume * c, q_top, q_wall, 0.0, 0.0],
        )

        # port labels
        self.assertEqual(SaltCrucible.output_port_labels["flux_top"], 2)
        self.assertEqual(SaltCrucible.output_port_labels["released_wall"], 5)

    def test_no_irradiation(self):
        """Without irradiation nothing is produced and nothing is released"""

        B = self.make(irradiations=None)
        _, y = self.simulate(B, 100.0, dt=1.0)
        np.testing.assert_array_equal(y, np.zeros_like(y))

    def test_no_irradiation_decay(self):
        """Without a source the initial inventory is released exponentially"""

        c0 = 1e12
        B = self.make(irradiations=None, initial_concentration=c0)
        rate = (B.A_top * B.k_top + B.A_wall * B.k_wall) / B.volume
        t, y = self.simulate(B, 3 / rate, tolerance_lte_rel=1e-9, dt=1.0)

        np.testing.assert_allclose(y[0], c0 * np.exp(-rate * t), rtol=1e-6)
        total = B.volume * c0 * (1 - np.exp(-rate * t))
        np.testing.assert_allclose(y[4] + y[5], total, rtol=1e-6, atol=1e-6 * total[-1])

    def test_constant_source(self):
        """Test against the analytic solution for a source that is always on

        With S constant and rate = (A_top k_top + A_wall k_wall) / V

            c(t) = S / (V rate) (1 - exp(-rate t))
            N_i(t) = A_i k_i S / (V rate) (t - (1 - exp(-rate t)) / rate)
        """

        B = self.make()
        S = B.TBR * B.neutron_rate
        rate = (B.A_top * B.k_top + B.A_wall * B.k_wall) / B.volume

        for Solver in [RKCK54, ESDIRK43]:
            with self.subTest(Solver=Solver.__name__):
                B = self.make()
                t, y = self.simulate(
                    B, 2 / rate, Solver=Solver, tolerance_lte_rel=1e-9, dt=1.0
                )

                c = S / (B.volume * rate) * (1 - np.exp(-rate * t))
                drift = t - (1 - np.exp(-rate * t)) / rate
                n_top = B.A_top * B.k_top * S / (B.volume * rate) * drift
                n_wall = B.A_wall * B.k_wall * S / (B.volume * rate) * drift

                np.testing.assert_allclose(y[0], c, rtol=1e-6, atol=1e-6 * c[-1])
                np.testing.assert_allclose(
                    y[4], n_top, rtol=1e-6, atol=1e-6 * n_top[-1]
                )
                np.testing.assert_allclose(
                    y[5], n_wall, rtol=1e-6, atol=1e-6 * n_wall[-1]
                )

                # the inventory output is volume times concentration
                np.testing.assert_allclose(y[1], B.volume * y[0])

        # steady state release equals production
        B = self.make()
        _, y = self.simulate(B, 30 / rate, tolerance_lte_rel=1e-9, dt=1.0)
        np.testing.assert_allclose(y[2][-1] + y[3][-1], S, rtol=1e-6)

        # split between the surfaces is the ratio of the transfer rates
        np.testing.assert_allclose(
            y[2][-1] / y[3][-1], (B.A_top * B.k_top) / (B.A_wall * B.k_wall), rtol=1e-9
        )

    def test_irradiation_schedule(self):
        """Test two irradiations and the decay in between against the exact solution"""

        B = self.make(irradiations=[(100.0, 400.0), (600.0, 700.0)], k_top=1e-4)
        rate = (B.A_top * B.k_top + B.A_wall * B.k_wall) / B.volume
        self.assertGreater(rate * 100, 0.1)  # the release is fast enough to see

        t, y = self.simulate(B, 1000.0, tolerance_lte_rel=1e-9, dt=1.0)
        exact = exact_release(B, t)

        for i, j in [(0, 0), (4, 1), (5, 2)]:
            np.testing.assert_allclose(
                y[i], exact[j], rtol=1e-6, atol=1e-6 * exact[j].max()
            )

        # nothing before the first irradiation
        self.assertTrue(np.all(y[:, t <= 100.0] == 0.0))

    def test_irradiation_edges_adaptive(self):
        """Adaptive solvers step exactly to the edges of the irradiations"""

        B = self.make(irradiations=[(100.3, 400.7)])
        t, _ = self.simulate(B, 500.0, Solver=RKBS32, tolerance_lte_rel=1e-6, dt=10.0)

        for edge in (100.3, 400.7):
            self.assertAlmostEqual(np.min(np.abs(t - edge)), 0.0, places=6)

    def test_adjacent_irradiations(self):
        """An irradiation that starts when the previous one stops keeps the source on"""

        B = self.make(irradiations=[(0.0, 200.0), (200.0, 400.0)], k_top=1e-4)
        ref = self.make(irradiations=[(0.0, 400.0)], k_top=1e-4)

        _, y = self.simulate(B, 600.0, tolerance_lte_rel=1e-9, dt=1.0)
        _, y_ref = self.simulate(ref, 600.0, tolerance_lte_rel=1e-9, dt=1.0)

        np.testing.assert_allclose(y[:, -1], y_ref[:, -1], rtol=1e-6)

    def test_fixed_step(self):
        """Fixed steps apply the switch at the end of the step that contains it"""

        B = self.make(irradiations=[(100.0, 400.0)], k_top=1e-4)
        t, y = self.simulate(B, 600.0, Solver=SSPRK22, dt=0.1)
        exact = exact_release(B, t[-1:])

        np.testing.assert_allclose(y[4][-1], exact[1][0], rtol=1e-3)
        np.testing.assert_allclose(y[5][-1], exact[2][0], rtol=1e-3)

    def test_mass_balance(self):
        """Produced tritium is either in the salt or released"""

        B = self.make(irradiations=[(0.0, 300.0), (500.0, 800.0)], k_top=1e-4)
        S = B.TBR * B.neutron_rate
        t, y = self.simulate(B, 1000.0, tolerance_lte_rel=1e-9, dt=1.0)

        produced = S * (np.minimum(t, 300.0) + np.clip(t - 500.0, 0.0, 300.0))
        accounted = y[1] + y[4] + y[5]
        np.testing.assert_allclose(
            accounted, produced, rtol=1e-6, atol=1e-6 * produced[-1]
        )

    def test_time_dependent_coefficients(self):
        """Test mass transport coefficients and neutron rate that depend on time"""

        def k_top(t):
            return 1e-5 * (1.0 + 0.5 * np.sin(t / 50.0))

        def k_wall(t):
            return 2e-6 * np.exp(-t / 400.0)

        def rate(t):
            return 1e8 * (1.0 + 0.2 * np.cos(t / 80.0))

        B = self.make(
            k_top=k_top, k_wall=k_wall, neutron_rate=rate, irradiations=[(0.0, 300.0)]
        )

        # independent high accuracy reference solution
        def rhs(t, x):
            S = B.TBR * rate(t) * float(t < 300.0)
            q_top = B.A_top * k_top(t) * x[0]
            q_wall = B.A_wall * k_wall(t) * x[0]
            return [(S - q_top - q_wall) / B.volume, q_top, q_wall]

        for Solver in [RKCK54, ESDIRK43]:
            with self.subTest(Solver=Solver.__name__):
                B = self.make(
                    k_top=k_top,
                    k_wall=k_wall,
                    neutron_rate=rate,
                    irradiations=[(0.0, 300.0)],
                )
                _, y = self.simulate(
                    B, 600.0, Solver=Solver, tolerance_lte_rel=1e-9, dt=1.0
                )

                x300 = solve_ivp(
                    rhs, (0, 300.0), [0, 0, 0], method="DOP853", rtol=1e-12, atol=1e-6
                ).y[:, -1]
                final = solve_ivp(
                    rhs, (300.0, 600.0), x300, method="DOP853", rtol=1e-12, atol=1e-6
                ).y[:, -1]

                np.testing.assert_allclose(y[0][-1], final[0], rtol=1e-6)
                np.testing.assert_allclose(y[4][-1], final[1], rtol=1e-6)
                np.testing.assert_allclose(y[5][-1], final[2], rtol=1e-6)

    def test_reset(self):
        """Reset restores the state and the source switch"""

        B = self.make(irradiations=[(0.0, 100.0)], k_top=1e-4)
        sco = Scope()
        sim = Simulation(
            [B, sco],
            [Connection(B["released_top"], sco)],
            Solver=RKCK54,
            tolerance_lte_rel=1e-9,
            dt=1.0,
            log=False,
        )

        sim.run(300.0)
        first = sco.read()[1][0][-1]

        sim.reset()
        sco.reset()
        sim.run(300.0)
        second = sco.read()[1][0][-1]

        self.assertGreater(first, 0.0)
        self.assertAlmostEqual(first, second, delta=1e-9 * first)


class TestBaby1LRun1(unittest.TestCase):
    """
    Reproduce the cumulative tritium release of the 0D model of BABY-1L run 1
    (LIBRA-project/BABY-1L-run-1, analysis/tritium/tritium_model.py), as
    computed by `libra_toolbox.tritium.model.Model`.
    """

    def run_release(self, times):
        B = run1_crucible()
        sim = Simulation(
            [B],
            [],
            Solver=RKCK54,
            tolerance_lte_rel=1e-9,
            tolerance_lte_abs=1e-6,
            dt=100.0,
            log=False,
        )
        top, wall = [], []
        for _ in run_to(sim, times):
            top.append(B.outputs["released_top"])
            wall.append(B.outputs["released_wall"])
        return B, np.array(top), np.array(wall)

    def test_geometry(self):
        """Volume of 1.22 L from 1.88 kg of salt"""

        B = run1_crucible()
        self.assertAlmostEqual(B.volume * 1e3, 1.22094, places=5)
        self.assertAlmostEqual(B.A_top, 0.0153938, places=7)
        self.assertAlmostEqual(B.A_wall, 0.0510373, places=7)

    def test_release_closed_form(self):
        """The block follows the closed form of the model at the sampling times"""

        times = np.sort(RUN1_TIMES_TOP)
        B, top, wall = self.run_release(times)
        exact = exact_release(B, times)

        np.testing.assert_allclose(top, exact[1], rtol=1e-8)
        np.testing.assert_allclose(wall, exact[2], rtol=1e-8)

    def test_release_vs_libra_toolbox(self):
        """Compare to the libra toolbox, which uses BDF with `rtol=1e-3`

        The toolbox undershoots the closed form by 0.76% at every time after
        the first irradiation, a solver error, so the tolerance is 1%.
        """

        _, top, wall = self.run_release(RUN1_TIMES_TOP)
        np.testing.assert_allclose(top, LIBRA_RELEASE_TOP, rtol=1e-2)
        np.testing.assert_allclose(wall[3:], LIBRA_RELEASE_WALL, rtol=1e-2)

        # the toolbox is below the block everywhere, not scattered around it
        self.assertTrue(np.all(LIBRA_RELEASE_TOP < top))
        self.assertTrue(np.all(LIBRA_RELEASE_WALL < wall[3:]))


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
