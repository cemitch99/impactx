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
kin_energy_MeV = 2000.0  # reference energy (kinetic)
bunch_charge_C = 25.0e-12  # used with space charge
npart = 100000  # number of macro particles

# set reference particle
ref = sim.beam.ref
ref.set_species("proton").set_kin_energy_MeV(kin_energy_MeV)

#   particle bunch
sigmaX = 0.003
sigmaY = sigmaX
sigmaPx = 0.002
sigmaPy = 0.002
sigmaT = 1.0e-4

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
    0.4,
    0.9,
    0.1,
)

sim.add_particles(bunch_charge_C, distr, npart, spin)

# design the accelerator lattice
ns = 10  # number of slices per ds in the element

# add beam diagnostics
monitor = elements.BeamMonitor("monitor", backend="h5")

gamma = ref.gamma
beta_gamma = ref.beta_gamma
lambda_c = 1.30692972

ks_value = 1.0e-7
bz_value = ks_value * beta_gamma

sigma_p = np.sqrt((sigmaPx * beta_gamma) ** 2 + (bz_value / 2.0) ** 2 * sigmaX**2)
bracket2 = (lambda_c / sigma_p + beta_gamma / (1 + gamma)) ** 2

ez_times_z = np.abs((gamma - 1.0 - bracket2 * (gamma + 1.0)) / (bracket2 - 1))
print("ez_times_z = ")
print(ez_times_z)

ds_value = 5.0
ez_value = ez_times_z / ds_value
print("ez = ")
print(ez_value)
print("ds = ")
print(ds_value)

sol_chr = elements.ChrAcc(
    name="sol_chr", ds=ds_value, ez=ez_value, bz=bz_value, nslice=ns
)

# set the lattice
sim.lattice.append(monitor)
sim.lattice.append(sol_chr)
sim.lattice.append(monitor)

# run simulation
sim.track_particles()

# clean shutdown
sim.finalize()
