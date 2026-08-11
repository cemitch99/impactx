/* Copyright 2022-2026 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Chad Mitchell, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_NONLINEARLENS_H
#define IMPACTX_NONLINEARLENS_H

#include "particles/ImpactXParticleContainer.H"
#include "mixin/alignment.H"
#include "mixin/beamoptic.H"
#include "mixin/thin.H"
#include "mixin/lineartransport.H"
#include "mixin/spintransport.H"
#include "mixin/named.H"
#include "mixin/nofinalize.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
// TODO: #include <AMReX_SIMD.H>
//   SIMD will require better complex math support in vir-simd / C++26
//   https://github.com/mattkretz/vir-simd/issues/42

#include <cmath>
#include <stdexcept>

namespace impactx::elements
{
    struct NonlinearLens
    : public mixin::Named,
      public mixin::BeamOptic<NonlinearLens>,
      public mixin::LinearTransport<NonlinearLens>,
      public mixin::Thin,
      public mixin::Alignment,
      public mixin::SpinTransport,
      public mixin::NoFinalize
      // TODO: public amrex::simd::Vectorized<amrex::simd::native_simd_size_particlereal>
    {
        static constexpr auto type = "NonlinearLens";
        using PType = ImpactXParticleContainer::ParticleType;

        /** Single short segment of the nonlinear magnetic insert element
         *
         *  A thin lens associated with a single short segment of the
         *  nonlinear magnetic insert described by V. Danilov and
         *  S. Nagaitsev, PRSTAB 13, 084002 (2010), Sect. V.A.  This
         *  element appears in MAD-X as type NLLENS.
         *
         * @param knll integrated strength of the nonlinear lens (m)
         * @param cnll distance of singularities from the origin (m)
         * @param dx horizontal translation error in m
         * @param dy vertical translation error in m
         * @param rotation_degree rotation error in the transverse plane [degrees]
         * @param name a user defined and not necessarily unique name of the element
         */
        NonlinearLens (
            amrex::ParticleReal knll,
            amrex::ParticleReal cnll,
            amrex::ParticleReal dx = DEFAULT_dx,
            amrex::ParticleReal dy = DEFAULT_dy,
            amrex::ParticleReal rotation_degree = DEFAULT_rotation_degree,
            std::optional<std::string> name = DEFAULT_name
        )
        : Named(std::move(name)),
          Alignment(dx, dy, rotation_degree),
          m_knll(knll), m_cnll(cnll)
        {
        }

        /** Reverse the element (negate strength) */
        void reverse () { m_knll = -m_knll; }

        /** Push all particles */
        using BeamOptic::operator();

        /** Compute and cache the constants for the push.
         *
         * In particular, used to pre-compute and cache variables that are
         * independent of the individually tracked particle.
         *
         * @param refpart reference particle
         */
        void compute_constants (RefPart const & refpart)
        {
            using namespace amrex::literals; // for _rt and _prt

            Alignment::compute_constants(refpart);

            m_inv_cnll = 1.0_prt / m_cnll;
            m_kick = -m_knll * m_inv_cnll;

            m_beta = refpart.beta();
            m_ibeta = 1_prt / m_beta;
        }

        /** This is a nonlinear lens functor, so that a variable of this type
         *  can be used like a nonlinear lens function.
         *
         * The @see compute_constants method must be called before pushing particles through this operator.
         *
         * @param x particle position in x
         * @param y particle position in y
         * @param t particle position in t (unused)
         * @param px particle momentum in x
         * @param py particle momentum in y
         * @param pt particle momentum in t (unused)
         * @param idcpu particle global index (unused)
         * @param refpart reference particle (unused)
         */
        template<typename T_Real=amrex::ParticleReal, typename T_IdCpu=uint64_t>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (
            T_Real const & AMREX_RESTRICT x,
            T_Real const & AMREX_RESTRICT y,
            [[maybe_unused]] T_Real const & AMREX_RESTRICT t,
            T_Real & AMREX_RESTRICT px,
            T_Real & AMREX_RESTRICT py,
            [[maybe_unused]] T_Real const & AMREX_RESTRICT pt,
            [[maybe_unused]] T_IdCpu const & AMREX_RESTRICT idcpu,
            [[maybe_unused]] RefPart const & AMREX_RESTRICT refpart
        ) const
        {
            using namespace amrex::literals; // for _rt and _prt
            using amrex::Math::powi;

            // a complex type with two T_Real
            using Complex = amrex::GpuComplex<T_Real>;

            // access reference particle values to find (beta*gamma)^2
            //amrex::ParticleReal const pt_ref = refpart.pt;
            //amrex::ParticleReal const betgam2 = powi<2>(pt_ref) - 1.0_prt;

            // initialize output values
            // T_Real xout = x;
            // T_Real yout = y;
            // T_Real tout = t;
            T_Real pxout = px;
            T_Real pyout = py;
            // T_Real ptout = pt;

            // assign complex position zeta = (x + iy)/cnll
            Complex zeta(x, y);
            zeta = zeta * m_inv_cnll;
            Complex const re1(1.0_prt, 0.0_prt);
            Complex const im1(0.0_prt, 1.0_prt);

            // compute croot = sqrt(1-zeta**2)
            Complex croot = powi<2>(zeta);
            croot = re1 - croot;
            croot = amrex::sqrt(croot);

            // compute carcsin = arcsin(zeta)
            Complex carcsin = im1 * zeta + croot;
            carcsin = -im1 * amrex::log(carcsin);

            // compute complex function F'(zeta)
            Complex dF = zeta / powi<2>(croot);
            dF = dF + carcsin / powi<3>(croot);

            // compute momentum kick
            T_Real const dpx = +m_kick * dF.m_real;
            T_Real const dpy = -m_kick * dF.m_imag;

            // advance position and momentum
            // xout = x;
            pxout = px + dpx;

            // yout = y;
            pyout = py + dpy;

            // tout = t;
            // ptout = pt;

            // assign updated values
            // x = xout;
            // y = yout;
            // t = tout;
            px = pxout;
            py = pyout;
            // pt = ptout;
        }

        /** This pushes the reference particle. */
        using Thin::operator();


        /** This operator pushes the particle spin.
         *
         * @param x particle position in x
         * @param y particle position in y
         * @param t particle position in t
         * @param px particle momentum in x
         * @param py particle momentum in y
         * @param pt particle momentum in t
         * @param sx particle spin x-component
         * @param sy particle spin y-component
         * @param sz particle spin z-component
         * @param idcpu particle global index
         * @param refpart reference particle (unused)
         */
        template<typename T_Real=amrex::ParticleReal, typename T_IdCpu=uint64_t>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void spin_and_phasespace_push (
            T_Real const & AMREX_RESTRICT x,
            T_Real const & AMREX_RESTRICT y,
            [[maybe_unused]] T_Real const & AMREX_RESTRICT t,
            T_Real & AMREX_RESTRICT px,
            T_Real & AMREX_RESTRICT py,
            [[maybe_unused]] T_Real const & AMREX_RESTRICT pt,
            T_Real & AMREX_RESTRICT sx,
            T_Real & AMREX_RESTRICT sy,
            T_Real & AMREX_RESTRICT sz,
            [[maybe_unused]] T_IdCpu const & AMREX_RESTRICT idcpu,
            [[maybe_unused]] RefPart const & AMREX_RESTRICT refpart
        ) const
        {
            using namespace amrex::literals; // for _rt and _prt
            using amrex::Math::powi;

            // reference particle relativistic factors
            amrex::ParticleReal const gamma_ref = refpart.gamma();
            amrex::ParticleReal const gyro_anomaly = refpart.gyromagnetic_anomaly;

            // initialize the three components of the axis-angle vector
            T_Real lambdax = 0_prt;
            T_Real lambday = 0_prt;
            T_Real lambdaz = 0_prt;

            // a complex type with two T_Real
            using Complex = amrex::GpuComplex<T_Real>;

            // assign complex position zeta = (x + iy)/cnll
            Complex zeta(x, y);
            zeta = zeta * m_inv_cnll;
            Complex const re1(1.0_prt, 0.0_prt);
            Complex const im1(0.0_prt, 1.0_prt);

            // compute croot = sqrt(1-zeta**2)
            Complex croot = powi<2>(zeta);
            croot = re1 - croot;
            croot = amrex::sqrt(croot);

            // compute carcsin = arcsin(zeta)
            Complex carcsin = im1 * zeta + croot;
            carcsin = -im1 * amrex::log(carcsin);

            // compute complex function F'(zeta)
            Complex dF = zeta / powi<2>(croot);
            dF = dF + carcsin / powi<3>(croot);

            // Integrated magnetic field normalized by q/mc (half-step):
            T_Real const Bx = -0.5_prt * m_kick * dF.m_imag * m_beta * gamma_ref;
            T_Real const By = -0.5_prt * m_kick * dF.m_real * m_beta * gamma_ref;
            T_Real const Bz = 0_prt;

            // Integrated electric field normalized by q/mc^2 (half-step):
            T_Real const Ex = 0.0_prt;
            T_Real const Ey = 0.0_prt;
            T_Real const Ez = 0.0_prt;

            // Quantities required to evaluate the full Thomas-BMT precession vector (before phase space kick).
            T_Real const Pnorm = sqrt(1_prt - 2_prt * pt * m_ibeta + pt*pt);
            T_Real const iPnorm = 1_prt/Pnorm;
            T_Real const gamma = gamma_ref * (1_prt - pt*m_beta);
            T_Real Ps = sqrt(Pnorm*Pnorm - px*px - py*py);
            T_Real ux = px * iPnorm;
            T_Real uy = py * iPnorm;
            T_Real uz = Ps * iPnorm;

            // Zero curvature in a quadrupole
            amrex::ParticleReal const h = 0.0_prt;

            // Evaluation of the full Thomas-BMT precession vector.
            tbmt_precession_vector(x,ux,uy,uz,gamma,h,gyro_anomaly,Bx,By,Bz,Ex,Ey,Ez,lambdax,lambday,lambdaz);

            // push the spin vector using the generator just determined (half spin-kick)
            rotate_spin(lambdax, lambday, lambdaz, sx, sy, sz);

            // phase space push
            (*this)(x, y, t, px, py, pt, idcpu, refpart);

            // Quantities required to evaluate the full Thomas-BMT precession vector (after phase space kick).
            Ps = sqrt(Pnorm*Pnorm - px*px - py*py);
            ux = px * iPnorm;
            uy = py * iPnorm;
            uz = Ps * iPnorm;

            // Evaluation of the full Thomas-BMT precession vector.
            tbmt_precession_vector(x,ux,uy,uz,gamma,h,gyro_anomaly,Bx,By,Bz,Ex,Ey,Ez,lambdax,lambday,lambdaz);

            // push the spin vector using the generator just determined (half spin-kick)
            rotate_spin(lambdax, lambday, lambdaz, sx, sy, sz);

        }


        /** This pushes the covariance matrix. */
        using LinearTransport::operator();

        /** This function returns the linear transport map.
         *
         * The Danilov-Nagaitsev kick implemented by @see operator() is
         *   dpx - i*dpy = -(knll / cnll) * F'(zeta),
         * with zeta = (x + i*y) / cnll and
         *   F'(zeta) = zeta / (1 - zeta^2) + arcsin(zeta) / (1 - zeta^2)^(3/2).
         * The Taylor series around zeta = 0 starts at
         *   F'(zeta) = 2*zeta + (8/3)*zeta^3 + O(zeta^5),
         * so the linearized map at the reference trajectory is a thin
         * quadrupole with equal and opposite focusing strengths in the
         * two transverse planes:
         *   R(2,1) = -2 * knll / cnll^2,
         *   R(4,3) = +2 * knll / cnll^2.
         *
         * Ideal behavior only: alignment and rotation errors, as well as
         * feed-down from an off-axis reference orbit, are ignored; the
         * map is evaluated at (x, y) = (0, 0).
         *
         * This element appears in MAD-X as type NLLENS. The linear limit
         * above is the MAD-X linear quadrupole equivalent of the NLLENS
         * thin kick (cf. MAD-X Physical Methods Manual, nonlinear lens).
         *
         * Reference: V. Danilov and S. Nagaitsev, Phys. Rev. ST Accel.
         * Beams 13, 084002 (2010), Sect. V.A
         *
         * @param[in] refpart reference particle (unused)
         * @returns 6x6 transport matrix
         */
        AMREX_GPU_HOST AMREX_FORCE_INLINE
        Map6x6
        transport_map ([[maybe_unused]] RefPart const & AMREX_RESTRICT refpart) const
        {
            using namespace amrex::literals; // for _rt and _prt
            using amrex::Math::powi;

            Map6x6 R = Map6x6::Identity();
            // Leading 2*zeta term of F'(zeta); higher orders vanish at zero.
            amrex::ParticleReal const k_lin = 2.0_prt * m_knll / powi<2>(m_cnll);
            R(2,1) = -k_lin;
            R(4,3) = +k_lin;

            // apply the transverse rotation (roll) alignment error
            return rotate_aligned_map(R);
        }

        amrex::ParticleReal m_knll; //! integrated strength of the nonlinear lens (m)
        amrex::ParticleReal m_cnll; //! distance of singularities from the origin (m)

      private:
        // constants that are independent of the individually tracked particle,
        // see: compute_constants() to refresh
        amrex::ParticleReal m_inv_cnll;  //! 1 / m_cnll
        amrex::ParticleReal m_kick;
        amrex::ParticleReal m_beta, m_ibeta;
    };

} // namespace impactx

IMPACTX_PUSH_EXTERN_TEMPLATE(impactx::elements::NonlinearLens)

#endif // IMPACTX_NONLINEARLENS_H
