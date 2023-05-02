# QSM Forward Model

Based on Marques, J. P., et al. (2021). QSM reconstruction challenge 2.0: A realistic in silico head phantom for MRI data simulation and evaluation of susceptibility mapping procedures. Magnetic Resonance in Medicine, 86(1), 526-542. https://doi.org/10.1002/mrm.28716

Includes code for:

 - Field model (forward multiplication with dipole kernel based on chi)
 - Signal model (magnitude and phase simulation based on field/M0/R1/R2star)
 - Phase offset model
 - Noise model
 - Shim field model
 - k-space cropping

See `main.py` for a full example.
