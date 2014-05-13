import h5py
from functools import reduce
import operator
from scipy import optimize
import numpy as np

from plot_images import draw
import matplotlib.pyplot as plt

input_file_name = "S00718_S00757.hdf5"

with h5py.File(input_file_name, "r") as input_file:
    phase_stepping_curves = input_file[
        "postprocessing/phase_stepping_curves"][...]
    phase_steps = phase_stepping_curves.shape[-1]
    original_shape = phase_stepping_curves.shape
    n_curves = reduce(operator.mul, original_shape[:-1], 1)
    curves = np.reshape(phase_stepping_curves, (n_curves, original_shape[-1]))
    flats = input_file["postprocessing/flat_parameters"][...]
    flats[..., 0] = flats[..., 0] / phase_steps
    flats[..., 2] = flats[..., 2] / phase_steps
    flats = np.reshape(flats, (n_curves, 3))
    visibility = input_file["postprocessing/visibility"][...]
    visibilities = np.reshape(visibility, n_curves)
    R = 0.45
    v_min = 0.02
    x_data = 2 * np.pi * np.arange(phase_steps) / phase_steps
    assert (phase_stepping_curves == np.reshape(curves, original_shape)).all()
    fixed_ratio_reconstruction = np.reshape(
        np.zeros_like(
            input_file["postprocessing/flat_parameters"][...]),
        (n_curves, 3))

    def fixed_ratio_signals(curve, flat, n_periods=1):
        f1 = flat[2]
        f0 = flat[0]
        phi_flat = flat[1]
        exponent = 1 + 1 / R
        def f(x, a, phi):
            a1 = a ** exponent * f1 / f0 ** exponent
            return a + a1 * np.sin(x / n_periods + phi)
        try:
            popt, pcov = optimize.curve_fit(
                f, x_data, curve, p0=(f0, 0))
            a = popt[0]
            a1 = a ** exponent * f1 / f0 ** exponent
            phi = popt[1] - phi_flat
        except RuntimeError:
            fourier_transformed = np.fft.fft(curve) / phase_steps
            a = np.abs(fourier_transformed[0])
            a1 = np.abs(fourier_transformed[n_periods]) / f1 * f0 / a
            phi = np.angle(fourier_transformed[n_periods]) - phi_flat
        phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
        return(a / f0, phi, a1)
    for i, (flat, curve) in enumerate(zip(flats, curves)):
        reconstruction = fixed_ratio_signals(curve, flat)
        fixed_ratio_reconstruction[i] = reconstruction

    fixed_ratio_reconstruction = np.reshape(
        fixed_ratio_reconstruction, original_shape[:-1] + (3,))

    height = 6
    draw(input_file_name, height,
         fixed_ratio_reconstruction[0, 400:500, ..., 0],
         fixed_ratio_reconstruction[0, 400:500, ..., 1],
         fixed_ratio_reconstruction[0, 400:500, ..., 2],
         )
