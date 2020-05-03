import numpy as np
from scipy.constants import electron_mass, elementary_charge, c, hbar, h
from pathlib import Path
import scipy.special

#from electropy.units import eV_to_Joule, invm_to_invÅ
#from electropy.momentum import 
pi = np.pi
bohr = 0.529  # Å, bohr_radius


def lorentz_factor(v):
    from scipy.constants import c
    import numpy as np

    return 1 / np.sqrt(1 - v ** 2 / c ** 2)


def gamma(v):
    return lorentz_factor(v)


def free_electron_plasmon_energy(N, V):
    """Calculates free electron plasmon energy Ep from the electron density in a unit cell

    E_{p,F} = \hbar\omega_{p} = \hbar\sqrt{\frac{N}{V(x))}\frac{e^{2}}{m_{0}\epsilon_{0}}}

    Parameters
    ----------
    N
        Valence electrons per unit cell.
    V
        Volume of the unit cell.

    Returns
    -------
    float | array of floats
        Plasmon energy in eV
    """
    import numpy as np
    from scipy.constants import epsilon_0, electron_mass, electron_volt, hbar

    density = N / V
    plasmon = (
        hbar * np.sqrt(density * electron_volt ** 2 / (electron_mass * epsilon_0))
    ) / electron_volt
    return plasmon


def semi_free_electron_plasmon_energy(N, V, Eg):
    """Calculates semi-free electron plasmon energy Ep from the electron density in a unit cell

    E_{p,sF} = \sqrt{E_{p,F}^{2} + E_{g}^{2}}

    Parameters
    ----------
    N
        Valence electrons per unit cell.
    V
        Volume of the unit cell.

    Returns
    -------
    float | array of floats
        Plasmon energy in eV

    """
    import numpy as np

    free = free_electron_plasmon_energy(N, V)
    semifree = np.sqrt(free ** 2 + Eg ** 2)
    return semifree

