import h5py
import numpy

with h5py.File("S00718_S00757.hdf5", "r") as input_file:
    phase_stepping_curves = input_file["postprocessing/phase_stepping_curves"]
    print(phase_stepping_curves.shape)
