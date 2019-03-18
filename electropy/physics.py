def relativistic_wavelength(kV=300):
    'Returns relativistic wavelength in meters'
    import numpy as np
    from scipy.constants import h, c, electron_mass, e
    V = 1e3*kV
    top = h*c
    bottom = np.sqrt(e*V*(2*electron_mass*c**2 + e*V))
    wavelength = top / bottom
    return wavelength


def relativistic_wavelength_with_potential(kV, Vs=0):
    import numpy as np
    from scipy.constants import h, c, electron_mass, e
    V = 1e3*kV
    top = h*c
    bottom = np.sqrt((e*V + e*Vs)*(2*electron_mass*c**2 + e*V + e*Vs))
    wavelength = top / bottom
    return wavelength


def lorentz_factor(v):
    from scipy.constants import c
    import numpy as np
    return 1/np.sqrt(1 - v**2/c**2)


def gamma(v):
    return lorentz_factor(v)


def classical_velocity(keV):
    import numpy as np
    from scipy.constants import electron_mass, e
    velocity = np.sqrt(2*e*1000*keV/electron_mass)
    return velocity


def relativistic_velocity(keV):
    import numpy as np
    from scipy.constants import c, electron_mass, e
    V = keV*1000
    return c * np.sqrt(1 - 1/(1+V*e/(electron_mass*c**2))**2)


def characteristic_scattering_angle(E_edge, keV):
    '''Edge in eV, beam energy in keV
    From Egerton: Electron Energy Loss Spectroscopy in the TEM, page 5,
    Rep. Prog. Phys. 72 (2009) 016502 (25pp)

    '''
    from scipy.constants import c, electron_mass, e
    v = relativistic_velocity(keV)
    T = keV*1000
    edge = E_edge
    theta_E = edge / (2*gamma(v)*T)
    return theta_E


def scherzer_defocus(kV=300, Cs_mm=0.001, nD=1):
    import numpy as np
    Cs = 1e-3*Cs_mm
    defocus = -np.sqrt((2*nD - 0.5)*Cs*relativistic_wavelength(kV))
    return defocus


def scherzer_aperture(kV=300, Cs_mm=0.001):
    Cs = 1e-3*Cs_mm
    wavelength = relativistic_wavelength(kV)
    kmax = (6/(Cs*wavelength**3))**(1/4)
    amax = wavelength*kmax
    return amax


def resolution_at_scherzer(kV=300, Cs_mm=0.001):
    Cs = 1e-3*Cs_mm
    wavelength = relativistic_wavelength(kV)
    kmax = (6/(Cs*wavelength**3))**(1/4)
    return 1/kmax


def invnm_to_mrad(k, kV=300):
    '''Converts reciprocal nm units to mrad units

    Sep 2018: Realised that there was an unnecessary 
    factor of 2 inside the arctan.
    '''
    import numpy as np
    wavelength = relativistic_wavelength(kV)
    k = k * 1 / (1e-9)
    theta = np.arctan(k*wavelength)
    return theta * 1000


def invnm_to_invÅ(invnm):
    return invnm/10


def invÅ_to_invnm(invÅ):
    return invÅ*10


def invÅ_to_mrad(k, kV=300):
    '''Converts reciprocal Å units to mrad units

    Sep 2018: Realised that there was an unnecessary
    factor of 2 inside the arctan.
    '''
    import numpy as np
    wavelength = relativistic_wavelength(kV)
    k = k * 1 / (1e-10)
    theta = np.arctan(k*wavelength)
    return theta * 1000


def mrad_to_invnm(mrad, kV=300):
    '''Converts mrad units to reciprocal nm units

    Sep 2018: Realised that there was an unnecessary 
    factor of 2 inside the arctan.
    '''
    import numpy as np
    theta = mrad / 1000
    wavelength = relativistic_wavelength(kV)
    k = np.tan(theta)/wavelength
    k = k * 1e-9
    return k


def mrad_to_invÅ(mrad, kV=300):
    '''Converts mrad units to reciprocal Å units

    Sep 2018: Realised that there was an unnecessary
    factor of 2 inside the arctan.
    '''
    import numpy as np
    theta = mrad / 1000
    wavelength = relativistic_wavelength(kV)
    k = np.tan(theta)/wavelength
    k = k * 1e-10
    return k


def invm_to_invÅ(m):
    return m*1e-10


def q_parallel_invÅ(eV, keV):
    '''
    Calculate the magnitude of the beam-parallel component of the
    q-vector in inverse Å
    '''
    q = (eV*electron_volt)/(hbar*relativistic_velocity(keV))
    q_invÅ = (invm_to_invÅ(q))
    return q_invÅ


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
    density = N/V
    plasmon = (hbar * np.sqrt(density*electron_volt**2 /
                              (electron_mass*epsilon_0)))/electron_volt
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
    semifree = np.sqrt(free**2 + Eg**2)
    return semifree

def angular_resolution(sample_length_Å, kV=300):
    'Returns in radians, not mrad'
    sample_length = sample_length_Å * 1e-10
    wavelength = relativistic_wavelength(kV)
    reciprocal_resolution_k = 1/sample_length
    ang_res = wavelength*reciprocal_resolution_k
    return ang_res