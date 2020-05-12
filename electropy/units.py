<<<<<<< HEAD
import numpy as np
from electropy.beam import relativistic_velocity, relativistic_wavelength
from scipy.constants import electron_mass, elementary_charge, c, hbar, h
=======
from electropy.beam import relativistic_velocity, relativistic_wavelength
>>>>>>> 501e771dac9fc370e5eb9170df16125b3615d204


def invnm_to_mrad(k, kV=300):
    """Converts reciprocal nm units to mrad units

    Sep 2018: Realised that there was an unnecessary 
    factor of 2 inside the arctan.
    """
    import numpy as np

    wavelength = relativistic_wavelength(kV)
    k = k * 1 / (1e-9)
    theta = np.arctan(k * wavelength)
    return theta * 1000


def invnm_to_invÅ(invnm):
    return invnm / 10


def invÅ_to_invnm(invÅ):
    return invÅ * 10


def invÅ_to_mrad(k, kV=300):
    """Converts reciprocal Å units to mrad units

    Sep 2018: Realised that there was an unnecessary
    factor of 2 inside the arctan.
    """
    import numpy as np

    wavelength = relativistic_wavelength(kV)
    k = k * 1 / (1e-10)
    theta = np.arctan(k * wavelength)
    return theta * 1000


def mrad_to_invnm(mrad, kV=300):
    """Converts mrad units to reciprocal nm units

    Sep 2018: Realised that there was an unnecessary 
    factor of 2 inside the arctan.
    """
    import numpy as np

    theta = mrad / 1000
    wavelength = relativistic_wavelength(kV)
    k = np.tan(theta) / wavelength
    k = k * 1e-9
    return k


def mrad_to_invÅ(mrad, kV=300):
    """Converts mrad units to reciprocal Å units

    When measuring from 000 to a point on a diffraction pattern,
    the angle calculated is the semi-angle (half full angle)
    
    Sep 2018: Realised that there was an unnecessary
    factor of 2 inside the arctan.
    """
    import numpy as np

    theta = mrad / 1000
    wavelength = relativistic_wavelength(kV)
    k = np.tan(theta) / wavelength
    k = k * 1e-10
    return k


def invm_to_invÅ(m):
    return m * 1e-10


def invm_to_invnm(m):
    return m * 1e-9


def eV_to_Joule(eV):
    return eV * elementary_charge

def eV_to_wavelength(eV):
    'electronvolt to nanometer. Returns inf if eV <= 0'
    if np.isscalar(eV):
        if eV <= 0.0:
            return np.inf
    else:
        eV = np.array(eV)
        eV[eV <= 0.0] = -1
    wave = 1e9 * h*c / (elementary_charge*eV)
    wave[wave < 0.0] = np.inf
    return wave

def wavelength_to_eV(nm):
    'wavelength to electronvolt'
    return h*c / (elementary_charge*(nm*1e-9))

def k_vector(kV):
    E0 = eV_to_Joule(kV * 1000)
    return (2 * electron_mass * E0 / hbar ** 2) ** (1 / 2)
