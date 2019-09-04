from electropy.physics import relativistic_velocity


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


def k_vector(kV):
    E0 = eV_to_Joule(kV * 1000)
    return (2 * electron_mass * E0 / hbar ** 2) ** (1 / 2)
