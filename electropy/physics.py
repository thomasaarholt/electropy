import numpy as np
from scipy.constants import electron_mass, elementary_charge, c, hbar, h
import electropy.physics
from pathlib import Path
import scipy.special
pi = np.pi
bohr = 0.529  # Å, bohr_radius


def relativistic_wavelength(kV=300):
    'Returns relativistic wavelength in meters'
    import numpy as np
    V = 1e3*kV
    top = h*c
    bottom = np.sqrt(
        elementary_charge*V*(
            2*electron_mass*c**2 + elementary_charge*V))
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


def invm_to_invnm(m):
    return m*1e-9


def q_parallel_invÅ(eV, keV):
    '''
    Calculate the magnitude of the beam-parallel component of the
    q-vector in inverse Å
    '''
    q = (eV*elementary_charge)/(hbar*relativistic_velocity(keV))
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


# Bear in mind that convention is Z, Y, X for indicies, not X, Y, Z
wavelength = relativistic_wavelength()
z = 0
k_z = 1/wavelength


def wavefunction(z):
    return np.exp(2*pi*1j*z/wavelength)


def transmission_function(x, sigma, Vz):
    return np.exp(1j*sigma*Vz(x))


def wavefunction_transmitted(x, sigma, Vz, z):
    t = transmission_function(x, sigma, Vz)
    return t * wavefunction(z)


def interaction_parameter(kV):
    wavelength = relativistic_wavelength(kV)
    velocity = relativistic_velocity(kV)
    return 2*pi*gamma(velocity)*electron_mass*elementary_charge*wavelength / h**2


def projected_potential(Vs, slice_thickness):
    return Vs*slice_thickness


def potential_fast(scattering, n, r):
    '''
    Calculates the potential very quickly, but may use too much memory. Add as an if statement for small memory cases.
    '''
    a, b, c, d = np.swapaxes(np.take(scattering, n-1, axis=0), 0, 1)
    r = r[..., None]
    left = 2*pi**2*bohr*elementary_charge*a/r*np.exp(-2*pi*r*np.sqrt(b))
    right = 2*pi**(5/2)*bohr*elementary_charge*c * \
        d**(-3/2)*np.exp(-pi**2*r**2/d)
    V = left + right
    return V.sum(axis=(-1, -2)).T


def projected_potential_fast(scattering, n, r):
    a, b, c, d = np.swapaxes(np.take(scattering, n-1, axis=0), 0, 1)
    r = r[..., None]

    left = 4*pi**2*bohr*elementary_charge*a*mod_bessel_zero(2*pi*b**0.5)
    right = 2*pi**2*bohr*elementary_charge*c/d*np.exp(-pi**2*r**2/d)
    pot_proj = left + right
    return pot_proj.sum(axis=(-1, -2)).T


def get_radius2D(cell, potential_spacing=0.05):
    shape = np.round(cell.cell.diagonal()[:2] / potential_spacing).astype(int)
    I = np.indices((shape).astype(int), dtype="uint16")
    I2 = np.stack(len(cell.positions)*[I])
    diff = (I2.T - cell.positions.T[:2]/potential_spacing)
    diff = diff.astype('float32')
    R = np.linalg.norm(diff, axis=-2)*potential_spacing
    # Radius smaller than 0.1Å set to 0.Å to avoid singularity (infinity)
    R[R < 0.1] = 0.1
    return R


def get_radius3D(cell, potential_spacing=0.05):
    shape = np.round(cell.cell.diagonal() / potential_spacing).astype(int)
    I1 = np.indices((shape).astype(int), dtype="uint16")
    I2 = np.stack(len(cell.positions)*[I1])
    diff = (I2.T - cell.positions.T/potential_spacing)
    diff = diff.astype('float32')
    R = np.linalg.norm(diff, axis=-2)*potential_spacing
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
    a, b, c, d = scattering[n-1]
    a/(q**2 + b) + c*np.exp(-d*q**2)


def load_scattering_matrix():
    with open(
        Path(
            electropy.physics.__file__).parent / "hartreefock.txt") as f:
        lines = f.readlines()
    scattering = np.zeros((len(lines), 4, 3), dtype="float32")
    for i, line in enumerate(lines):
        data = line.split('chisq= ')[1].strip().split(' ')[1:]
        data = [float(d) for d in data]
        scattering[i] = np.array(data, dtype='float32').reshape((3, 4)).T
    return scattering


def mod_bessel_zero(x):
    return scipy.special.kn(0, x)


def eV_to_Joule(eV):
    return eV*elementary_charge


def k_vector(kV):
    E0 = eV_to_Joule(kV*1000)
    return (2*electron_mass*E0 / hbar**2)**(1/2)


# def q_perpendicular(invnm, kV):
#     '''Returns in invnm
#     '''
#     k = k_vector(kV)
#     theta = invnm_to_mrad(invnm)/1000
#     qpr_invnm = invm_to_invnm(k*theta)
#     return qpr_invnm


def q_perpendicular(mrad, kV):
    '''Returns in invnm

    Most accurate form would be:
    (2*electron_mass*elementary_charge*
    (keV*1000-dE))**(0.5)/hbar*sin(theta) * 1e-9
    '''
    theta = mrad/1000
    k = k_vector(kV)
    qpr_invnm = invm_to_invnm(k*theta)
    return qpr_invnm


def q_parallel(dE, kV):
    '''dE in eV
    Returns in invnm
    Alternative formula: qll = k0*thetaE
    '''
    dE = eV_to_Joule(dE)
    y = gamma(relativistic_velocity(kV))
    E0 = eV_to_Joule(kV*1000)
    qll = (electron_mass)**0.5*dE / (y*hbar*(2*E0)**0.5)
    qll_invnm = invm_to_invnm(qll)
    return qll_invnm


def q_parallel_alt(dE, kV):
    '''dE om eV
    Returns in invnm
    '''
    k = k_vector(kV)
    thetaE = characteristic_scattering_angle(dE, kV)
    qll_invnm = invm_to_invnm(k*thetaE)
    return qll_invnm


def q_total(q_perp, dE, kV):
    return (q_parallel(dE, kV)**2 + q_perp**2)**0.5
