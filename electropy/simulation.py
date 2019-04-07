import numpy as np
from pathlib import Path
import os
from numba import jit, prange
from scipy.constants import electron_mass, elementary_charge, c, Planck

bohr = 0.529  # Å, bohr_radius
pi = np.pi


def read_element_scattering_factors():
    scattering_data_file = Path(os.path.dirname(__file__)) / "hartreefock.txt"

    with open(scattering_data_file) as f:
        lines = f.readlines()
    scattering = np.zeros((len(lines), 4, 3), dtype="float32")
    for i, line in enumerate(lines):
        data = line.split('chisq= ')[1].strip().split(' ')[1:]
        data = [float(d) for d in data]
        scattering[i] = np.array(data, dtype='float32').reshape((3, 4)).T
    return scattering


@jit(nopython=True)
def potential_left(r, a, b):
    return 2*pi**2*bohr*elementary_charge*a/r*np.exp(-2*pi*r*np.sqrt(b))


@jit(nopython=True)
def potential_right(r, c, d):
    return 2*pi**(5/2)*bohr*elementary_charge*c*d**(-3/2)*np.exp(-pi**2*r**2/d)


def potential_vector(n, r, scattering):
    '''
    Calculates the potential very quickly, but may use too much memory.
    Add an if statement for small memory cases.
    '''
    a, b, c, d = np.swapaxes(np.take(scattering, n-1, axis=0), 0, 1)
    r = r[..., None]
    V = potential_left(r, a, b) + potential_right(r, c, d)
    return V


@jit(nopython=True)
def potential_loop(n, r, scattering):
    a = scattering[n-1][0]
    b = scattering[n-1][1]
    c = scattering[n-1][2]
    d = scattering[n-1][3]
    V = potential_left(r, a, b) + potential_right(r, c, d)
    return V.sum()


def scattering_amplitude(q, n, scattering):
    a, b, c, d = scattering[n-1]
    return a/(q**2 + b) + c*np.exp(-d*q**2)


@jit(nopython=True)
def norm(x, y, z):
    return (x**2 + y**2 + z**2)**0.5


def calculate_potential_loop(cell, potential_spacing=0.1):
    shape = np.round(cell.cell.diagonal() / potential_spacing).astype(int)
    scattering = read_element_scattering_factors()
    pot = np.zeros(shape)
    return loop_potential_loop(
        cell.positions, cell.numbers, pot, potential_spacing, scattering)


@jit(nopython=True, parallel=True)
def loop_potential_loop(
        positions, numbers, pot, potential_spacing, scattering):
    for i in prange(len(numbers)):
        for (x1, y1, z1), val in np.ndenumerate(pot):
            # x1 is current index
            x2, y2, z2 = positions[i] / potential_spacing
            x, y, z = x1 - x2, y1 - y2, z1 - z2
            r = norm(x, y, z)*potential_spacing
            r = 0.1 if r < 0.1 else r
            pot[x1, y1, z1] += potential_loop(numbers[i], r, scattering)
    return pot


def calculate_potential_vector(cell, potential_spacing=0.1):
    shape = np.round(cell.cell.diagonal() / potential_spacing).astype(int)
    pot = np.zeros(shape)
    scattering = read_element_scattering_factors()
    index_array = np.indices(shape.astype(int), dtype="uint16")
    index_array = np.stack(len(cell)*[index_array])  # One for each atom
    distance_from_nuclei = (
        index_array.T - cell.positions.T/potential_spacing).astype('float32')
    radius = np.linalg.norm(distance_from_nuclei, axis=-2)*potential_spacing

    radius[radius < 0.1] = 0.1

    atomic_numbers = cell.numbers.T

    pot = potential_vector(atomic_numbers, radius,
                           scattering).sum(axis=(-1, -2)).T
    return pot
