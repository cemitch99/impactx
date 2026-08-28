/* Copyright 2022-2026 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Alex Bojanich, Chad Mitchell, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_PARTICLES_WAKEFIELDS_SHORTRANGEWAKEELEMENT_H_
#define IMPACTX_PARTICLES_WAKEFIELDS_SHORTRANGEWAKEELEMENT_H_

#include "elements/RFCavity.H"
#include "elements/ShortRF.H"
#include "elements/Buncher.H"
#include "elements/ChrUniformAcc.H"
#include "particles/ReferenceParticle.H"

#include <AMReX_REAL.H>

#include <cmath>
#include <tuple>
#include <variant>


namespace impactx::particles::wakefields
{
    /** Function to check if an element supports short range structure wakefields and to calculate parameters needed for wake.
     *
     * @param[in] element_variant the lattice element to check
     * @param[in] refpart reference particle
     * @returns boolean flag indicating if the element supports short range wakefields
     */
    template <typename VariantType>
    std::tuple<bool, std::vector<amrex::ParticleReal>, std::vector<amrex::ParticleReal>, std::vector<amrex::ParticleReal>, std::vector<amrex::ParticleReal>>
    ShortRangeWakeElement (
        VariantType const& element_variant,
        [[maybe_unused]] RefPart const & refpart
    )
    {
        std::vector<amrex::ParticleReal> z_data;
        std::vector<amrex::ParticleReal> wake_x_data;
        std::vector<amrex::ParticleReal> wake_y_data;
        std::vector<amrex::ParticleReal> wake_z_data;
        bool element_has_short_range_wake = false;

        if (std::holds_alternative<elements::RFCavity>(element_variant))
        {
            // Access element parameters as needed.
            auto& element = std::get<elements::RFCavity>(element_variant);
            z_data = elements::RFCavity::WakeData::get(element.m_wake_id)->z_data.host_const();
            wake_x_data = elements::RFCavity::WakeData::get(element.m_wake_id)->wake_x_data.host_const();
            wake_y_data = elements::RFCavity::WakeData::get(element.m_wake_id)->wake_y_data.host_const();
            wake_z_data = elements::RFCavity::WakeData::get(element.m_wake_id)->wake_z_data.host_const();
            element_has_short_range_wake = true;
        }
        else if (std::holds_alternative<elements::ShortRF>(element_variant))
        {
            // Access element parameters as needed.
            // auto& element = std::get<elements::ShortRF>(element_variant);
            element_has_short_range_wake = true;
        }
        else if (std::holds_alternative<elements::Buncher>(element_variant))
        {
            // Access element parameters as needed.
            // auto& element = std::get<elements::Buncher>(element_variant);
            element_has_short_range_wake = true;
        }
        else if (std::holds_alternative<elements::ChrAcc>(element_variant))
        {
            // Access element parameters as needed.
            // auto& element = std::get<elements::ChrAcc>(element_variant);
            element_has_short_range_wake = true;
        }

        return {element_has_short_range_wake, z_data, wake_x_data, wake_y_data, wake_z_data};
    }

} // namespace impactx::particles::wakefields

#endif // IMPACTX_PARTICLES_WAKEFIELDS_SHORTRANGEWAKEELEMENT_H_
