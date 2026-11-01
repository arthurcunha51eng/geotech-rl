import math

import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from geotech_rl import physics


def by_integration(L, D, e, phi, gamma, c, n):
    """The published integral form of the rotation model, as a check on the closed form."""
    kp = physics.passive_coefficient(phi)
    p = lambda z: n * D * (kp * gamma * z + 2.0 * c * math.sqrt(kp))

    def moment(zr):
        return quad(lambda z: p(z) * (e + z), 0.0, zr)[0] - quad(lambda z: p(z) * (e + z), zr, L)[0]

    zr = brentq(moment, 1e-9, L, xtol=1e-12)
    return quad(p, 0.0, zr)[0] - quad(p, zr, L)[0]


@pytest.mark.parametrize("L,D,e,phi,gamma,c,n", [
    (2.2, 0.58, 6.5, 33.0, 18.76, 0.0, 1.5),
    (1.5, 0.40, 7.2, 33.0, 16.8, 6.0, 1.5),
    (2.6, 0.70, 8.2, 36.0, 18.0, 10.0, 2.0),
])
def test_closed_form_matches_integration(L, D, e, phi, gamma, c, n):
    assert physics.ultimate_load(L, D, e, phi, gamma, c, n) == pytest.approx(by_integration(L, D, e, phi, gamma, c, n), rel=1e-7)


def test_suction_adds_capacity():
    assert physics.ultimate_load(2.0, 0.5, 6.7, 33.0, 17.0, 6.0) > physics.ultimate_load(2.0, 0.5, 6.7, 33.0, 17.0, 0.0)
