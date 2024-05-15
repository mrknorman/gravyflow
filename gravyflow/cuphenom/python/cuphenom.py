import numpy as np
from ctypes import *
import os
import logging

NUM_POLARISATION_STATES : int = 2

def to_ctypes(to_convert):
    return (c_float * len(to_convert))(*to_convert)

def ensure_list(item):
    return item if isinstance(item, (list, tuple, set, np.ndarray)) else [item]

# Load the library
current_dir = os.path.dirname(os.path.realpath(__file__))
lib = CDLL(f'{current_dir}/libphenom.so')

lib.pythonWrapperPhenomD.argtypes = \
[
    c_int,    # Num Waveforms
    c_float,  # sample_rate_hertz
    c_float,  # duration_seconds
    POINTER(c_float),  # mass_1_msun
    POINTER(c_float),  # mass_2_msun
    POINTER(c_float),  # inclination_radians
    POINTER(c_float),  # distance_mpc
    POINTER(c_float),  # reference_orbital_phase_in
    POINTER(c_float),  # ascending_node_longitude
    POINTER(c_float),  # eccentricity
    POINTER(c_float),  # mean_periastron_anomaly
    POINTER(c_float),  # spin_1_in
    POINTER(c_float)  # spin_2_in
]
lib.pythonWrapperPhenomD.restype = POINTER(c_float)

def imrphenomd(
    num_waveforms = 1,
    sample_rate_hertz = 4096.0,
	duration_seconds = 2.0,
	mass_1_msun = 30.0,
	mass_2_msun = 30.0,
	inclination_radians = 0.0,
	distance_mpc = 1000.0,
	reference_orbital_phase_in = 0.0,
	ascending_node_longitude = 0.0,
	eccentricity = 0.0,
	mean_periastron_anomaly = 0.0,
	spin_1_in = [0.0, 0.0, 0.0],
	spin_2_in = [0.0, 0.0, 0.0]
	):
    
    args = locals().copy()
    
    length_one_args = ['num_waveforms', 'sample_rate_hertz', 'duration_seconds']
    length_three_args = ['spin_1_in', 'spin_2_in']
    
    args = {name: ensure_list(val) for name, val in args.items()}
            
    # Check that all arguments have length num_waveforms
    args_to_check = {
        name: val for name, val in args.items()
        if name not in length_one_args + length_three_args
    }
    
    for arg_name, arg_value in args_to_check.items():
        assert len(arg_value) == num_waveforms, \
        (f"{arg_name} does not have the expected length of {num_waveforms}, "
         "instead = {len(arg_value)}")
    
    # Ensure input spins are float arrays of length 3
    assert len(spin_1_in) == len(spin_2_in) == 3*num_waveforms
    
    # Convert all arguments except the first three to ctypes
    ctypes_args = {
        name: to_ctypes(val) for name, val in args.items()
        if name not in length_one_args
    }

    # Call the C function
    waveform_pointer = lib.pythonWrapperPhenomD(
        num_waveforms,
        sample_rate_hertz,
        duration_seconds,
        ctypes_args["mass_1_msun"],
        ctypes_args["mass_2_msun"],
        ctypes_args["inclination_radians"],
        ctypes_args["distance_mpc"],
        ctypes_args["reference_orbital_phase_in"],
        ctypes_args["ascending_node_longitude"],
        ctypes_args["eccentricity"],
        ctypes_args["mean_periastron_anomaly"],
        ctypes_args["spin_1_in"],
        ctypes_args["spin_2_in"]
    )

    total_num_samples = \
        int(np.floor(sample_rate_hertz * duration_seconds))*num_waveforms
    
    waveforms = np.ctypeslib.as_array(
        waveform_pointer, shape=(total_num_samples * 2,)
    )
    
    # Reshape array:
    num_samples = int(sample_rate_hertz*duration_seconds)
    strain = np.stack((
        waveforms[::2].reshape(-1, num_samples),
        waveforms[1::2].reshape(-1, num_samples)
    ), axis=1)

    return strain
    
    