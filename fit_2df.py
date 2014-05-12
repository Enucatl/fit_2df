import h5py
from functools import reduce
import operator
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("S00718_S00757.hdf5", "r") as input_file:
    phase_stepping_curves = input_file["postprocessing/phase_stepping_curves"][...]
    phase_steps = phase_stepping_curves.shape[-1]
    original_shape = phase_stepping_curves.shape
    n_curves = reduce(operator.mul, original_shape[:-1], 1)
    curves = np.reshape(phase_stepping_curves, (n_curves, original_shape[-1]))
    flat_1 = input_file["postprocessing/flat_parameters"][..., 2] / phase_steps
    flat_0 = input_file["postprocessing/flat_parameters"][..., 0] / phase_steps
    f1s = np.reshape(flat_1, n_curves)
    f0s = np.reshape(flat_0, n_curves)
    visibility = input_file["postprocessing/visibility"][...]
    visibilities = np.reshape(visibility, n_curves)
    R = 0.45
    v_min = 0.02
    x_data = 2 * np.pi * np.arange(phase_steps) / phase_steps
    assert (phase_stepping_curves == np.reshape(curves, original_shape)).all()
    assert (flat_1 == np.reshape(f1s, original_shape[:-1])).all()
    assert (flat_0 == np.reshape(f0s, original_shape[:-1])).all()
    for v, f1, f0, curve in zip(visibilities, f1s, f0s, curves):
        if v > v_min:
            def f(x, a, phi):
                exponent = 1 + 1 / R
                a1 = a ** exponent * f1 / f0 ** exponent
                return a + a1 * np.sin(x + phi)
            popt, pcov = optimize.curve_fit(
                f, x_data, curve, p0=(f0, 0))
            print(v, popt)
        else:
            print(None)
