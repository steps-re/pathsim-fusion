#########################################################################################
##
##                                 Salt Crucible Block
##
#########################################################################################

# IMPORTS ===============================================================================

from typing import ClassVar

import numpy as np
from pathsim.blocks.dynsys import DynamicalSystem
from pathsim.events.schedule import ScheduleList

# HELPERS ===============================================================================


def _as_function(value):
    """Wrap a constant into a function of time, pass callables through."""
    if callable(value):
        return value
    return lambda t: value


# BLOCK DEFINITIONS =====================================================================


class SaltCrucible(DynamicalSystem):
    """Zero dimensional tritium release from a salt filled crucible.

    The salt is a single well mixed volume. Tritium is bred in the salt while
    the neutron generator is on and leaves through two surfaces, the free top
    surface (towards the sweep gas) and the wall. This is the model of
    ``libra_toolbox.tritium.model.Model`` used for the BABY and FLiBe-1L
    experiments, written as a block so that it can be connected to the
    sampling and measurement blocks.


    Mathematical Formulation
    -------------------------
    The salt concentration :math:`c` and the cumulative releases
    :math:`N_{top}` and :math:`N_{wall}` are the states of the block:

    .. math::

        \\begin{align}
            V \\frac{dc}{dt} &= S(t) - Q_{top} - Q_{wall} \\\\
            \\frac{dN_{top}}{dt} &= Q_{top} = A_{top} \\, k_{top}(t) \\, c \\\\
            \\frac{dN_{wall}}{dt} &= Q_{wall} = A_{wall} \\, k_{wall}(t) \\, c
        \\end{align}

    The source is the tritium breeding ratio times the neutron rate, active
    only inside the irradiation windows:

    .. math::

        S(t) = \\begin{cases}
            \\mathrm{TBR} \\cdot \\Gamma_n(t) & , t \\in [t_{start}, t_{stop}) \\\\
            0 & , \\text{otherwise}
        \\end{cases}

    The external concentration is neglected, as in the reference model.

    The geometry is a cylinder of radius :math:`r` and salt height :math:`h`
    with

    .. math::

        V = \\pi r^2 h, \\quad
        A_{top} = \\pi r^2, \\quad
        A_{wall} = 2 \\pi (r + L_{wall}) h + A_{top}

    where the bottom disc is counted as wall (same as the reference model).

    Note
    ----
    The irradiation windows are resolved by internal scheduled events, so
    adaptive solvers step exactly to the start and stop of an irradiation. With
    a fixed timestep the switch is applied at the end of the step that contains
    it. The coefficients ``k_top``, ``k_wall`` and ``neutron_rate`` may be
    functions of time, for example to model a mass transfer coefficient that
    changes during a run.

    Units are not enforced. In SI (m, m², m³, s) the concentration is in
    particles/m³ and the fluxes in particles/s when ``neutron_rate`` is in
    neutrons/s and ``TBR`` in particles/neutron. Multiply a flux with the decay
    constant to get an activity in Bq.

    Parameters
    ----------
    radius : float
        radius of the salt cylinder [m]
    height : float
        height of the salt column [m]
    TBR : float
        tritium breeding ratio [particles/neutron]
    neutron_rate : float | callable
        neutron rate [neutrons/s], constant or function of time
    k_top : float | callable
        mass transport coefficient of the top surface [m/s], constant or
        function of time
    k_wall : float | callable
        mass transport coefficient of the wall [m/s], constant or function of
        time
    irradiations : list[tuple[float, float]]
        ``(start, stop)`` times of the irradiations [s]. ``stop`` may be
        ``numpy.inf``. Default None (no irradiation).
    wall_thickness : float
        wall thickness added to the radius for the wall area [m]. Default is
        0.06 inch, as in the reference model.
    initial_concentration : float
        salt concentration at t = 0 [particles/m³]. Default 0.

    Attributes
    ----------
    volume : float
        salt volume [m³]
    A_top : float
        top surface area [m²]
    A_wall : float
        wall area including the bottom [m²]

    Outputs
    -------
    concentration (0) : tritium concentration in the salt
    inventory (1) : tritium inventory in the salt, volume times concentration
    flux_top (2) : release rate through the top surface
    flux_wall (3) : release rate through the wall
    released_top (4) : cumulative release through the top surface
    released_wall (5) : cumulative release through the wall
    """

    input_port_labels: ClassVar[dict[str, int]] = {}

    output_port_labels: ClassVar[dict[str, int]] = {
        "concentration": 0,
        "inventory": 1,
        "flux_top": 2,
        "flux_wall": 3,
        "released_top": 4,
        "released_wall": 5,
    }

    def __init__(
        self,
        radius,
        height,
        TBR,
        neutron_rate,
        k_top,
        k_wall,
        irradiations=None,
        wall_thickness=0.06 * 0.0254,
        initial_concentration=0.0,
    ):

        # input validation
        if radius <= 0:
            raise ValueError(f"'radius' must be positive but is {radius}")
        if height <= 0:
            raise ValueError(f"'height' must be positive but is {height}")
        if TBR < 0:
            raise ValueError(f"'TBR' must be non-negative but is {TBR}")
        if wall_thickness < 0:
            raise ValueError(
                f"'wall_thickness' must be non-negative but is {wall_thickness}"
            )
        for name, value in (
            ("neutron_rate", neutron_rate),
            ("k_top", k_top),
            ("k_wall", k_wall),
        ):
            if not callable(value) and value < 0:
                raise ValueError(f"'{name}' must be non-negative but is {value}")

        irradiations = [] if irradiations is None else [tuple(i) for i in irradiations]
        for start, stop in irradiations:
            if not stop > start:
                raise ValueError(
                    f"irradiation stop must be after start but is ({start}, {stop})"
                )

        # parameters
        self.radius = radius
        self.height = height
        self.TBR = TBR
        self.neutron_rate = neutron_rate
        self.k_top = k_top
        self.k_wall = k_wall
        self.irradiations = irradiations
        self.wall_thickness = wall_thickness
        self.initial_concentration = initial_concentration

        # source switch, toggled by the internal events
        self._irradiating = self._in_irradiation(0.0)

        # dynamical component, ode rhs for [c, N_top, N_wall]
        def _fn_d(x, u, t):
            q_top, q_wall = self._release_rates(x[0], t)
            dc = (self._source(t) - q_top - q_wall) / self.volume
            return np.array([dc, q_top, q_wall])

        # jacobian of the rhs, linear in the state
        def _jac_d(x, u, t):
            a_top = self.A_top * _as_function(self.k_top)(t)
            a_wall = self.A_wall * _as_function(self.k_wall)(t)
            return np.array(
                [
                    [-(a_top + a_wall) / self.volume, 0.0, 0.0],
                    [a_top, 0.0, 0.0],
                    [a_wall, 0.0, 0.0],
                ]
            )

        # algebraic output component
        def _fn_a(x, u, t):
            q_top, q_wall = self._release_rates(x[0], t)
            return np.array(
                [x[0], self.volume * x[0], q_top, q_wall, x[1], x[2]],
            )

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=np.array([initial_concentration, 0.0, 0.0]),
            jac_dyn=_jac_d,
        )

        self._create_irradiation_events()

    # geometry --------------------------------------------------------------------

    @classmethod
    def from_salt_mass(cls, salt_mass, salt_density, radius, **kwargs):
        """Create the crucible with the salt height set by the salt mass.

        Parameters
        ----------
        salt_mass : float
            mass of the salt [kg]
        salt_density : float
            density of the salt at operating temperature [kg/m³]
        radius : float
            radius of the salt cylinder [m]
        **kwargs
            all other arguments of the constructor
        """
        if salt_mass <= 0 or salt_density <= 0:
            raise ValueError("'salt_mass' and 'salt_density' must be positive")
        height = salt_mass / salt_density / (np.pi * radius**2)
        return cls(radius=radius, height=height, **kwargs)

    @property
    def volume(self):
        """Salt volume [m³]"""
        return self.A_top * self.height

    @property
    def A_top(self):
        """Top surface area [m²], neglects any re-entrant heater"""
        return np.pi * self.radius**2

    @property
    def A_wall(self):
        """Wall area including the bottom disc [m²]"""
        perimeter = 2 * np.pi * (self.radius + self.wall_thickness)
        return perimeter * self.height + self.A_top

    # model terms -----------------------------------------------------------------

    def _in_irradiation(self, t):
        return any(start <= t < stop for start, stop in self.irradiations)

    def _source(self, t):
        """Tritium production rate at time t."""
        if not self._irradiating:
            return 0.0
        return self.TBR * _as_function(self.neutron_rate)(t)

    def _release_rates(self, c, t):
        """Release rate through the top and the wall for concentration c."""
        q_top = self.A_top * _as_function(self.k_top)(t) * c
        q_wall = self.A_wall * _as_function(self.k_wall)(t) * c
        return q_top, q_wall

    # irradiation events ----------------------------------------------------------

    def _create_irradiation_events(self):
        """Create one event at every start and stop of an irradiation that
        switches the source on or off.
        """

        def switch_source(t):
            self._irradiating = self._in_irradiation(t)

        # a stop can coincide with the next start, so one event per unique time
        times = sorted(
            {float(t) for window in self.irradiations for t in window if 0 < t < np.inf}
        )

        self.events = (
            [ScheduleList(times_evt=times, func_act=switch_source)] if times else []
        )

    def reset(self):
        """Reset the block and the source switch."""
        super().reset()
        self._irradiating = self._in_irradiation(0.0)
