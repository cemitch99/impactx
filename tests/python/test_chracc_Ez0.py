#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import numpy as np

import amrex.space3d as amr
from impactx import ImpactX, elements


def test_element_push():
    """
    This tests using ImpactX without a lattice.
    """
    sim = ImpactX()

    # set numerical parameters and IO control
    sim.space_charge = False
    sim.slice_step_diagnostics = True

    # domain decomposition & space charge mesh
    sim.init_grids()

    # basic beam parameters
    kin_energy_MeV = 2000.0  # reference energy (kinetic)
    bunch_charge_C = 25.0e-12  # used with space charge

    # set reference particle
    ref = sim.beam.ref
    ref.set_species("proton").set_kin_energy_MeV(kin_energy_MeV)
    qm_eev = ref.charge_qe / (ref.mass_MeV * 1.0e6)  # electron charge/mass in e / eV

    # set test particles
    pc = sim.beam

    # npart = 50
    npart = 20

    ptmin = -3.0e-3
    ptmax = 3.0e-3

    #  add test particles
    if amr.ParallelDescriptor.IOProcessor():
        dpt = np.linspace(ptmin, ptmax, npart)
        zero_arr = np.linspace(0, 0.0, npart)
        pc.add_n_particles(
            zero_arr,
            zero_arr,
            zero_arr,
            zero_arr,
            zero_arr,
            dpt,
            qm_eev,
            bunch_charge=0.0,
        )

    # store initial particles
    df_initial = pc.to_df()

    # design the accelerator lattice
    ns = 1  # number of slices per ds in the element

    # add beam diagnostics
    monitor = elements.BeamMonitor("monitor", backend="h5")

    # bend radius (> 0)
    ks_value = 1.0

    # length for this test should be one period
    ds_value = 2.0 * np.pi / (ref.gyromagnetic_anomaly * ks_value)

    idrift_chr = elements.ChrDrift(name="idrift_chr", ds=-ds_value, nslice=ns)

    bz_value = ks_value * ref.beta_gamma
    ez_value = 0.0
    sol_chr = elements.ChrAcc(
        name="sol_chr", ds=ds_value, ez=ez_value, bz=bz_value, nslice=ns
    )

    # set the lattice
    sim.lattice.append(monitor)
    sim.lattice.append(sol_chr)
    sim.lattice.append(idrift_chr)
    sim.lattice.append(monitor)

    # run simulation
    sim.track_particles()

    pc = sim.beam
    df_final = pc.to_df()

    PHASE_COLS = [
        "position_x",
        "position_y",
        "position_t",
        "momentum_x",
        "momentum_y",
        "momentum_t",
    ]

    for c in PHASE_COLS:
        np.testing.assert_allclose(
            df_final[c].to_numpy(),
            df_initial[c].to_numpy(),
            atol=1.0e-12,
            rtol=0,
            err_msg=f"Roundtrip mismatch in {c}",
        )

    # clean shutdown
    sim.finalize()
