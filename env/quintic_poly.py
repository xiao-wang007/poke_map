"""
Quintic polynomial for smooth strike trajectory generation.

The smoothstep  f(u) = 6u^5 - 15u^4 + 10u^3  maps u in [0,1] to [0,1]
with zero first and second derivatives at both boundaries.
The contact point is at the midpoint: u=0.5 → s=0.5L.

Trajectory scaling:
    s(t) = L * f(t/T)           position  (m)
    v(t) = (L/T) * f'(t/T)      velocity  (m/s),  peak = 1.875 * L/T  at midpoint
    a(t) = (L/T^2) * f''(t/T)   accel     (m/s^2)

Trajectory duration from peak velocity v_mid:
    T = L * 1.875 / v_mid        (since f'(0.5) = 1.875)
"""


_PEAK_DERIVATIVE = 1.875  # f'(0.5) for the quintic smoothstep


def quintic_poly_query(u):
    """Position profile: s/L = f(u), range [0, 1]."""
    return 6 * u**5 - 15 * u**4 + 10 * u**3


def quintic_poly_derivative(u):
    """Velocity derivative: f'(u), peak = 1.875 at u=0.5."""
    return 30 * u**4 - 60 * u**3 + 30 * u**2


def quintic_poly_second_derivative(u):
    """Acceleration derivative: f''(u), zero at boundaries."""
    return 120 * u**3 - 180 * u**2 + 60 * u


def compute_T(v_mid, L):
    """Trajectory duration from peak velocity and travel distance."""
    if v_mid <= 0:
        return 1.0
    return (L * _PEAK_DERIVATIVE) / v_mid


def compute_strike_params(v_mid, L, dt, max_t, min_steps):
    """Compute (T, num_steps) clamped to sane bounds.

    Parameters
    ----------
    v_mid : float        Peak velocity at trajectory midpoint (m/s).
    L : float            Total strike distance (m).
    dt : float           Control period (1 / freq).
    max_t : float        Maximum allowed trajectory duration (s).
    min_steps : int      Minimum number of control steps.

    Returns
    -------
    T : float            Trajectory duration clamped to [min_steps*dt, max_t].
                         Tiny/non-positive v_mid uses max_t instead of a
                         short fallback, giving the slowest allowed strike.
    num_steps : int      Number of control steps, at least min_steps.
    """
    if v_mid <= 1e-6:
        T = max_t
    else:
        T = compute_T(v_mid, L)
    T = max(min_steps * dt, min(T, max_t))
    num_steps = max(min_steps, int(T / dt) + 1)
    return T, num_steps
