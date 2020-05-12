import numpy as np
from pathlib import Path
import scipy.special
import electropy.physics

from scipy.constants import pi, electron_mass, elementary_charge, c, hbar, h, physical_constants
from electropy.beam import relativistic_velocity, relativistic_wavelength
from electropy.physics import gamma

bohr = physical_constants['Bohr radius']

def projected_potential(Vs, slice_thickness):
    return Vs * slice_thickness


def potential_fast(scattering, n, r):
    """
    Calculates the potential very quickly, but may use too much memory. Add as an if statement for small memory cases.
    """
    a, b, c, d = np.swapaxes(np.take(scattering, n - 1, axis=0), 0, 1)
    r = r[..., None]
    left = (
        2
        * pi ** 2
        * bohr
        * elementary_charge
        * a
        / r
        * np.exp(-2 * pi * r * np.sqrt(b))
    )
    right = (
        2
        * pi ** (5 / 2)
        * bohr
        * elementary_charge
        * c
        * d ** (-3 / 2)
        * np.exp(-pi ** 2 * r ** 2 / d)
    )
    V = left + right
    return V.sum(axis=(-1, -2)).T


def projected_potential_fast(scattering, n, r):
    a, b, c, d = np.swapaxes(np.take(scattering, n - 1, axis=0), 0, 1)
    r = r[..., None]

    left = (
        4 * pi ** 2 * bohr * elementary_charge * a * mod_bessel_zero(2 * pi * b ** 0.5)
    )
    right = (
        2 * pi ** 2 * bohr * elementary_charge * c / d * np.exp(-pi ** 2 * r ** 2 / d)
    )
    pot_proj = left + right
    return pot_proj.sum(axis=(-1, -2)).T


def get_radius2D(cell, potential_spacing=0.05):
    shape = np.round(cell.cell.diagonal()[:2] / potential_spacing).astype(int)
    I = np.indices((shape).astype(int), dtype="uint16")
    I2 = np.stack(len(cell.positions) * [I])
    diff = I2.T - cell.positions.T[:2] / potential_spacing
    diff = diff.astype("float32")
    R = np.linalg.norm(diff, axis=-2) * potential_spacing
    # Radius smaller than 0.1Å set to 0.Å to avoid singularity (infinity)
    R[R < 0.1] = 0.1
    return R


def get_radius3D(cell, potential_spacing=0.05):
    shape = np.round(cell.cell.diagonal() / potential_spacing).astype(int)
    I1 = np.indices((shape).astype(int), dtype="uint16")
    I2 = np.stack(len(cell.positions) * [I1])
    diff = I2.T - cell.positions.T / potential_spacing
    diff = diff.astype("float32")
    R = np.linalg.norm(diff, axis=-2) * potential_spacing
    # Radius smaller than 0.1Å set to 0.Å to avoid singularity (infinity)
    R[R < 0.1] = 0.1
    return R


def potential_from_cell(cell, scattering, potential_spacing=0.05):
    R = get_radius3D(cell, potential_spacing)
    N = cell.numbers
    pot = potential_fast(scattering, N, R)
    return pot


def projected_potential_from_cell(cell, scattering, potential_spacing=0.05):
    R = get_radius2D(cell, potential_spacing)
    N = cell.numbers
    pot = projected_potential_fast(scattering, N, R)
    return pot


def scattering_amplitude(scattering, q, n):
    a, b, c, d = scattering[n - 1]
    a / (q ** 2 + b) + c * np.exp(-d * q ** 2)


def load_scattering_matrix():
    with open(Path(electropy.physics.__file__).parent / "hartreefock.txt") as f:
        lines = f.readlines()
    scattering = np.zeros((len(lines), 4, 3), dtype="float32")
    for i, line in enumerate(lines):
        data = line.split("chisq= ")[1].strip().split(" ")[1:]
        data = [float(d) for d in data]
        scattering[i] = np.array(data, dtype="float32").reshape((3, 4)).T
    return scattering


def mod_bessel_zero(x):
    return scipy.special.kn(0, x)


def interaction_parameter(kV):
    wavelength = relativistic_wavelength(kV)
    velocity = relativistic_velocity(kV)
    return (
        2
        * pi
        * gamma(velocity)
        * electron_mass
        * elementary_charge
        * wavelength
        / h ** 2
    )


def wavefunction_transmitted(x, sigma, Vz, z, wavelength):
    t = transmission_function(x, sigma, Vz)
    return t * wavefunction(z, wavelength)


def wavefunction(z, wavelength):
    return np.exp(2 * pi * 1j * z / wavelength)


def transmission_function(x, sigma, Vz):
    return np.exp(1j * sigma * Vz(x))

