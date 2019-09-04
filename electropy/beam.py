import numpy as np
from scipy.constants import electron_mass, elementary_charge, c, hbar, h


def relativistic_wavelength(kV=300):
    "Returns relativistic wavelength in meters"
    V = 1e3 * kV
    top = h * c
    bottom = np.sqrt(
        elementary_charge * V * (2 * electron_mass * c ** 2 + elementary_charge * V)
    )
    wavelength = top / bottom
    return wavelength


def relativistic_wavelength_with_potential(kV, Vs=0):
    import numpy as np
    from scipy.constants import h, c, electron_mass, e

    V = 1e3 * kV
    top = h * c
    bottom = np.sqrt((e * V + e * Vs) * (2 * electron_mass * c ** 2 + e * V + e * Vs))
    wavelength = top / bottom
    return wavelength


def classical_velocity(keV):
    import numpy as np
    from scipy.constants import electron_mass, e

    velocity = np.sqrt(2 * e * 1000 * keV / electron_mass)
    return velocity


def relativistic_velocity(keV):
    import numpy as np
    from scipy.constants import c, electron_mass, e

    V = keV * 1000
    return c * np.sqrt(1 - 1 / (1 + V * e / (electron_mass * c ** 2)) ** 2)


def scherzer_defocus(kV=300, Cs_mm=0.001, nD=1):
    import numpy as np

    Cs = 1e-3 * Cs_mm
    defocus = -np.sqrt((2 * nD - 0.5) * Cs * relativistic_wavelength(kV))
    return defocus


def scherzer_aperture(kV=300, Cs_mm=0.001):
    Cs = 1e-3 * Cs_mm
    wavelength = relativistic_wavelength(kV)
    kmax = (6 / (Cs * wavelength ** 3)) ** (1 / 4)
    amax = wavelength * kmax
    return amax


def resolution_at_scherzer(kV=300, Cs_mm=0.001):
    Cs = 1e-3 * Cs_mm
    wavelength = relativistic_wavelength(kV)
    kmax = (6 / (Cs * wavelength ** 3)) ** (1 / 4)
    return 1 / kmax


def optimal_convergence_angle_STEM(kV, Cs):
    "According to equation 2 in Weyland and Muller"
    wavelength = relativistic_wavelength(kV)
    return (4 * wavelength / Cs) ** (1 / 4)


def optimal_resolution_STEM(kV, Cs):
    wavelength = relativistic_wavelength(kV)
    return 0.43 * Cs ** (1 / 4) * wavelength ** (3 / 4)

