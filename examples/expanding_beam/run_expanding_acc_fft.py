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
sim.max_level = 1
sim.n_cell = [32, 32, 32]
sim.blocking_factor_x = [16]
sim.blocking_factor_y = [16]
sim.blocking_factor_z = [4]
# one box per level and process: this example runs on a single process
sim.max_grid_size_x = [64]
sim.max_grid_size_y = [64]
sim.max_grid_size_z = [64]

sim.particle_shape = 2  # B-spline order
sim.space_charge = "3D"
sim.poisson_solver = "fft"
sim.dynamic_size = True
sim.prob_relative = [1.2, 1.1]

# beam diagnostics
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# initial beam properties
kin_energy_MeV = 250  # reference energy
bunch_charge_C = 1.0e-9  # used with space charge
npart = 100000  # number of macro particles (outside tests, use 1e5 or more)

#   reference particle
ref = sim.beam.ref
ref.set_species("electron").set_kin_energy_MeV(kin_energy_MeV)
ref_for_envelope = ref.copy()

#   particle bunch
distr = distribution.Kurth6D(
    lambdaX=4.472135955e-4,
    lambdaY=4.472135955e-4,
    lambdaT=9.12241869e-7,
    lambdaPx=0.0,
    lambdaPy=0.0,
    lambdaPt=0.0,
)
sim.add_particles(bunch_charge_C, distr, npart)

# design the accelerator lattice
acc_section = elements.ChrAcc(name="acc_section", ds=6.0, ez=100.0, bz=0.0, nslice=100)
sim.lattice.extend([acc_section])

# run particle simulation
sim.track_particles()

# collect beam moments
rbc_particles = sim.beam.beam_moments()

# set up envelope simulation
sim.lattice.clear()
sim.lattice.extend([acc_section])

# run envelope simulation
sim.init_envelope(ref_for_envelope, distr, bunch_charge_C)
sim.track_envelope()

# collect beam moments
rbc_envelope = sim.envelope.beam_moments(ref_for_envelope)

sigx_part = rbc_particles["sigma_x"]
sigy_part = rbc_particles["sigma_y"]
sigt_part = rbc_particles["sigma_t"]

sigx_env = rbc_envelope["sigma_x"]
sigy_env = rbc_envelope["sigma_y"]
sigt_env = rbc_envelope["sigma_t"]

# clean shutdown
sim.finalize()

print("")

print("Particle Beam:")
print(f"  sigx={sigx_part:e} sigy={sigy_part:e} sigt={sigt_part:e}")

print("")
print("Envelope Beam:")
print(f"  sigx={sigx_env:e} sigy={sigy_env:e} sigt={sigt_env:e}")

atol = 0.0  # ignored
rtol = 1.5 * npart**-0.5  # from random sampling of a smooth distribution
print(f"  rtol={rtol} (ignored: atol~={atol})")

assert np.allclose(
    [sigx_part, sigy_part, sigt_part],
    [sigx_env, sigy_env, sigt_env],
    rtol=rtol,
    atol=atol,
)
