from physics import relativistic_velocity, eV_to_Joule
from electropy.units import k_vector, invm_to_invnm, eV_to_Joule

# def q_perpendicular(invnm, kV):
#     '''Returns in invnm
#     '''
#     k = k_vector(kV)
#     theta = invnm_to_mrad(invnm)/1000
#     qpr_invnm = invm_to_invnm(k*theta)
#     return qpr_invnm


def q_perpendicular(mrad, kV):
    """Returns in invnm

    Most accurate form would be:
    (2*electron_mass*elementary_charge*
    (keV*1000-dE))**(0.5)/hbar*sin(theta) * 1e-9
    """
    theta = mrad / 1000
    k = k_vector(kV)
    qpr_invnm = invm_to_invnm(k * theta)
    return qpr_invnm


def q_parallel(dE, kV):
    """dE in eV
    Returns in invnm
    Alternative formula: qll = k0*thetaE
    """
    dE = eV_to_Joule(dE)
    y = gamma(relativistic_velocity(kV))
    E0 = eV_to_Joule(kV * 1000)
    qll = (electron_mass) ** 0.5 * dE / (y * hbar * (2 * E0) ** 0.5)
    qll_invnm = invm_to_invnm(qll)
    return qll_invnm


def q_parallel_alt(dE, kV):
    """dE om eV
    Returns in invnm
    """
    k = k_vector(kV)
    thetaE = characteristic_scattering_angle(dE, kV)
    qll_invnm = invm_to_invnm(k * thetaE)
    return qll_invnm


def q_total(q_perp, dE, kV):
    return (q_parallel(dE, kV) ** 2 + q_perp ** 2) ** 0.5


def q_total_mrad(mrad, dE, kV):
    return (q_parallel(dE, kV) ** 2 + q_perpendicular(mrad, kV) ** 2) ** 0.5


def characteristic_scattering_angle(E_edge, keV):
    """Edge in eV, beam energy in keV
    From Egerton: Electron Energy Loss Spectroscopy in the TEM, page 5,
    Rep. Prog. Phys. 72 (2009) 016502 (25pp)

    """
    v = relativistic_velocity(keV)
    T = keV * 1000
    edge = E_edge
    theta_E = edge / (2 * gamma(v) * T)
    return theta_E


def q_parallel_invÅ(eV, keV):
    """
    Calculate the magnitude of the beam-parallel component of the
    q-vector in inverse Å
    """
    q = (eV * elementary_charge) / (hbar * relativistic_velocity(keV))
    q_invÅ = invm_to_invÅ(q)
    return q_invÅ
