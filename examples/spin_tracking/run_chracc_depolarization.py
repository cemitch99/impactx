#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np

from impactx import ImpactX, distribution, elements

sim = ImpactX()

# set numerical parameters and IO control
sim.space_charge = False
sim.spin = True
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# basic beam parameters
kin_energy_MeV = 20.00  # reference energy (kinetic)
bunch_charge_C = 25.0e-12  # used with space charge
npart = 1000000  # number of macro particles

# set reference particle
ref = sim.beam.ref
ref.set_species("proton").set_kin_energy_MeV(kin_energy_MeV)

#   particle bunch
sigmaX = 0.003
sigmaY = sigmaX
sigmaPx = 0.0
sigmaPy = 0.2
sigmaT = 0.0

polarization_x = 0.4
#polarization_y = 0.9
#polarization_z = 0.1
polarization_y = 0.8
polarization_z = 0.2

distr = distribution.Gaussian(
    lambdaX=sigmaX,
    lambdaY=sigmaY,
    lambdaT=sigmaT,
    lambdaPx=sigmaPx,
    lambdaPy=sigmaPy,
    lambdaPt=0.0,
    muxpx=0.0,
    muypy=0.0,
    mutpt=0.0,
)
spin = distribution.SpinvMF(
    polarization_x,
    polarization_y,
    polarization_z,
)

sim.add_particles(bunch_charge_C, distr, npart, spin)

# design the accelerator lattice
ns = 1  # number of slices per ds in the element

# add beam diagnostics
monitor = elements.BeamMonitor("monitor", backend="h5")

gamma = ref.gamma
beta_gamma = ref.beta_gamma

ks_value = 0.0
bz_value = ks_value * beta_gamma

ez_value = 10.0
ds_value = 50.0

sol_chr = elements.ChrAcc(
    name="sol_chr", ds=ds_value, ez=ez_value, bz=bz_value, nslice=ns
)

# set the lattice
sim.lattice.append(monitor)
sim.lattice.append(sol_chr)
sim.lattice.append(monitor)

# run simulation
sim.track_particles()

gamma_f = ref.gamma
beta_gamma_f = ref.beta_gamma

G = ref.gyromagnetic_anomaly

muterm = beta_gamma / (1 + gamma) - beta_gamma_f / (1 + gamma_f)
Gterm = -G * np.log((beta_gamma_f + gamma_f) / (beta_gamma + gamma))
lambda1 = np.abs(muterm + Gterm) * sigmaPy * beta_gamma

polarization_xf_pred = polarization_x
polarization_yf_pred = polarization_y * np.exp(-(lambda1**2) / 2.0)
polarization_zf_pred = polarization_z * np.exp(-(lambda1**2) / 2.0)

rbc = sim.beam.beam_moments()
polarization_xf = rbc["mean_sx"]
polarization_yf = rbc["mean_sy"]
polarization_zf = rbc["mean_sz"]

print("")
print("Final polarization predicted:")
print(
    f"  polarization_x={polarization_xf_pred:e} polarization_y={polarization_yf_pred:e} polarization_z={polarization_zf_pred:e}"
)
print("")
print("Final polarization:")
print(
    f"  polarization_x={polarization_xf:e} polarization_y={polarization_yf:e} polarization_z={polarization_zf:e}"
)

# clean shutdown
sim.finalize()

atol = 1.0e-3
assert np.allclose(
    [polarization_xf_pred, polarization_yf_pred, polarization_zf_pred],
    [polarization_xf, polarization_yf, polarization_zf],
    atol=atol,
)
