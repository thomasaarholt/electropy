import numpy as np


def read_element_scattering_factors():
    with open("hartreefock.txt") as f:
        lines = f.readlines()
    scattering = np.zeros((len(lines), 4, 3), dtype="float32")
    for i, line in enumerate(lines):
        data = line.split('chisq= ')[1].strip().split(' ')[1:]
        data = [float(d) for d in data]
        scattering[i] = np.array(data, dtype='float32').reshape((3, 4)).T
