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
sim.max_level = 0
sim.n_cell = [32, 32, 1]
sim.blocking_factor_x = [32]
sim.blocking_factor_y = [32]
sim.blocking_factor_z = [1]

sim.particle_shape = 2  # B-spline order
sim.space_charge = "2D"
sim.poisson_solver = "fft"
sim.dynamic_size = True
sim.prob_relative = [1.1]

# beam diagnostics
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = False

# domain decomposition & space charge mesh
sim.init_grids()

# initial beam properties
kin_energy_MeV = 250  # reference energy
beam_current_A = 0.15  # used with space charge
npart = 10000  # number of macro particles (outside tests, use 1e5 or more)

#   reference particle
ref = sim.beam.ref
ref.set_species("proton").set_kin_energy_MeV(kin_energy_MeV)
ref_for_envelope = ref.copy()

sig_xy_i = 5.0e-4
sig_t_i = 1.0e-3

#   particle bunch
distr = distribution.KVdist(
    lambdaX=sig_xy_i,
    lambdaY=sig_xy_i,
    lambdaT=sig_t_i,
    lambdaPx=0.0,
    lambdaPy=0.0,
    lambdaPt=0.0,
)
sim.add_particles(beam_current_A, distr, npart)

distance = 10.612823669911099

# design the accelerator lattice
acc_section = elements.ChrAcc(
    name="acc_section", ds=distance, ez=0.05, bz=0.0, nslice=100
)
sim.lattice.extend([acc_section])

# run particle simulation
sim.track_particles()

# collect beam moments
rbc_particles = sim.beam.beam_moments()

# set up envelope simulation
sim.lattice.clear()
sim.lattice.extend([acc_section])

# run envelope simulation
sim.init_envelope(ref_for_envelope, distr, beam_current_A)
sim.track_envelope()

# collect beam moments
rbc_envelope = sim.envelope.beam_moments(ref_for_envelope)

sigx_part = rbc_particles["sigma_x"]
sigy_part = rbc_particles["sigma_y"]
sigt_part = rbc_particles["sigma_t"]
emitx_part = rbc_particles["emittance_x"]
emity_part = rbc_particles["emittance_y"]
emitt_part = rbc_particles["emittance_t"]

sigx_env = rbc_envelope["sigma_x"]
sigy_env = rbc_envelope["sigma_y"]
sigt_env = rbc_envelope["sigma_t"]
emitx_env = rbc_envelope["emittance_x"]
emity_env = rbc_envelope["emittance_y"]
emitt_env = rbc_envelope["emittance_t"]

# clean shutdown
sim.finalize()

print("")

print("Predicted Expansion Factor:")
print(sigx_env / sig_xy_i)

print("")

print("Particle Beam:")
print(f"  sigx={sigx_part:e} sigy={sigy_part:e} sigt={sigt_part:e}")
print(
    f"  emittance_x={emitx_part:e} emittance_y={emity_part:e} emittance_t={emitt_part:e}"
)

print("")
print("Envelope Beam:")
print(f"  sigx={sigx_env:e} sigy={sigy_env:e} sigt={sigt_env:e}")
print(
    f"  emittance_x={emitx_env:e} emittance_y={emity_env:e} emittance_t={emitt_env:e}"
)

print("")
atol = 0.0  # ignored
rtol = 2.0 * npart**-0.5  # from random sampling of a smooth distribution
print(f"  rtol for beam size = {rtol} (ignored: atol~={atol})")

assert np.allclose(
    [sigx_part, sigy_part, sigt_part],
    [sigx_env, sigy_env, sigt_env],
    rtol=rtol,
    atol=atol,
)

atol = 3.0e-9
rtol = 0.0
print(f"  atol for emittances = {atol} (ignored: rtol~={rtol})")

assert np.allclose(
    [emitx_part, emity_part, emitt_part],
    [emitx_env, emity_env, emitt_env],
    rtol=rtol,
    atol=atol,
)
