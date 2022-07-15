/* Copyright 2022 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Axel Huebl, Chad Mitchell, Ji Qiang, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */
#include "ImpactX.H"
#include "initialization/InitOneBoxPerRank.H"
#include "particles/ImpactXParticleContainer.H"
#include "particles/Push.H"
#include "particles/transformation/CoordinateTransformation.H"
#include "particles/diagnostics/DiagnosticOutput.H"

#include <AMReX.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <memory>


namespace impactx
{
    ImpactX::ImpactX ()
        : AmrCore(initialization::one_box_per_rank()),
          m_particle_container(std::make_unique<ImpactXParticleContainer>(this))
    {
        // todo: if amr.n_cells is provided, overwrite/redefine AmrCore object

        // todo: if charge deposition and/or space charge are requested, require
        //       amr.n_cells from user inputs
    }

    void ImpactX::initGrids ()
    {
        // this is the earliest point that we need to know the particle shape,
        // so that we can initialize the guard size of our MultiFabs
        m_particle_container->SetParticleShape();

        // init blocks / grids & MultiFabs
        AmrCore::InitFromScratch(0.0);
        amrex::Print() << "boxArray(0) " << boxArray(0) << std::endl;

        // move old diagnostics out of the way
        amrex::UtilCreateCleanDirectory("diags", true);
    }

    void ImpactX::evolve (int num_steps)
    {
        BL_PROFILE("ImpactX::evolve");

        // print initial particle distribution to file
        diagnostics::DiagnosticOutput(*m_particle_container,
                                      diagnostics::OutputType::PrintParticles,
                                      "diags/initial_beam.txt");

        // print initial reference particle to file
        diagnostics::DiagnosticOutput(*m_particle_container,
                                      diagnostics::OutputType::PrintRefParticle,
                                      "diags/initial_ref_particle.txt");

        // print the initial values of the two invariants H and I
        diagnostics::DiagnosticOutput(*m_particle_container,
                                      diagnostics::OutputType::PrintNonlinearLensInvariants,
                                      "diags/initial_nonlinear_lens_invariants.txt");

        // loop over all beamline elements
        for (auto & element_variant : m_lattice)
        {
            // sub-steps for space charge within the element
            for (int step = 0; step < num_steps; ++step)
            {
                BL_PROFILE("ImpactX::evolve::step");
                amrex::Print() << " ++++ Starting step=" << step << "\n";

                // transform from x',y',t to x,y,z
                transformation::CoordinateTransformation(*m_particle_container,
                                                         transformation::Direction::T2Z);

                // Space-charge calculation: turn off if there is only 1 particle
                if (m_particle_container->TotalNumberOfParticles(false,false) > 1) {

                    // Note: The following operation assume that
                    // the particles are in x, y, z coordinates.

                    // Resize the mesh, based on `m_particle_container` extent
                    ResizeMesh();

                    // Redistribute particles in the new mesh in x, y, z
                    m_particle_container->Redistribute();

                    // charge deposition
                    m_particle_container->DepositCharge(m_rho, this->refRatio());

                    // poisson solve in x,y,z
                    //   TODO

                    // gather and space-charge push in x,y,z , assuming the space-charge
                    // field is the same before/after transformation
                    //   TODO
                }

                // transform from x,y,z to x',y',t
                transformation::CoordinateTransformation(*m_particle_container,
                                                         transformation::Direction::Z2T);

                // for later: original Impact implementation as an option
                // Redistribute particles in x',y',t
                //   TODO: only needed if we want to gather and push space charge
                //         in x',y',t
                //   TODO: change geometry beforehand according to transformation
                //m_particle_container->Redistribute();
                //
                // in original Impact, we gather and space-charge push in x',y',t ,
                // assuming that the distribution did not change

                // push all particles with external maps
                Push(*m_particle_container, element_variant);

                // just prints an empty newline at the end of the step
                amrex::Print() << "\n";

            } // end in-element space-charge sub-step loop
        } // end beamline element loop

        // print final particle distribution to file
        diagnostics::DiagnosticOutput(*m_particle_container,
                                      diagnostics::OutputType::PrintParticles,
                                      "diags/output_beam.txt");

        // print final reference particle to file
        diagnostics::DiagnosticOutput(*m_particle_container,
                                      diagnostics::OutputType::PrintRefParticle,
                                      "diags/output_ref_particle.txt");

        // print the final values of the two invariants H and I
        diagnostics::DiagnosticOutput(*m_particle_container,
                                      diagnostics::OutputType::PrintNonlinearLensInvariants,
                                      "diags/output_nonlinear_lens_invariants.txt");

    }
} // namespace impactx
