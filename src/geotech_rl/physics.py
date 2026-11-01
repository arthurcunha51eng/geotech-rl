"""Lateral capacity of a short rigid pole in sand.

The ground truth uses a rigid pole rotating about an interior point, with ultimate soil reaction per metre
p(z) = n D (Kp gamma z + 2 c sqrt(Kp)). c is the apparent cohesion that suction gives unsaturated sand, zero
when the sand is soaked. n is the field pressure coefficient of backfill compacted around a collar.

Lengths in m, forces in kN, unit weights in kN/m3, cohesion in kPa.
"""
import math

from scipy.optimize import brentq

G = 9.81


def passive_coefficient(phi_deg: float) -> float:
    return math.tan(math.radians(45.0 + phi_deg / 2.0)) ** 2


def unit_weight(dry_density: float, water_content: float) -> float:
    """Moist unit weight from dry density (g/cm3) and water content (fraction)."""
    return dry_density * (1.0 + water_content) * G


def ultimate_load(L: float, D: float, e: float, phi_deg: float, gamma: float, c: float = 0.0, n: float = 1.5) -> float:
    """Ultimate horizontal load at height e above ground, from moment and force equilibrium."""
    kp = passive_coefficient(phi_deg)
    a = n * D * kp * gamma
    b = n * D * 2.0 * c * math.sqrt(kp)

    def force(z):
        return a * z * z / 2.0 + b * z

    def moment(z):
        return a * (e * z * z / 2.0 + z ** 3 / 3.0) + b * (e * z + z * z / 2.0)

    total = moment(L)
    zr = brentq(lambda z: 2.0 * moment(z) - total, 1e-9, L, xtol=1e-12)
    return 2.0 * force(zr) - force(L)


def rotation_unit(L: float, D: float, e: float, gamma: float) -> float:
    """Rotation about an interior point, per unit of site constant: an independent shape for the same calibration."""
    return ultimate_load(L, D, e, 33.0, gamma, 0.0, 1.0)


def broms_unit(L: float, D: float, e: float, gamma: float) -> float:
    """Broms short rigid pile in cohesionless soil, rotation at the toe, per unit of Kp."""
    return 0.5 * gamma * D * L ** 3 / (e + L)
