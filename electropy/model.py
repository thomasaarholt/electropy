from electropy.beam import relativistic_wavelength
def angular_resolution(sample_length_Å, kV=300):
    "Returns in radians, not mrad"
    sample_length = sample_length_Å * 1e-10
    wavelength = relativistic_wavelength(kV)
    reciprocal_resolution_k = 1 / sample_length
    ang_res = wavelength * reciprocal_resolution_k
    return ang_res
