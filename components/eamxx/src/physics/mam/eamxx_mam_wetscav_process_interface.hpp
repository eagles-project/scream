#ifndef EAMXX_MAM_WETSCAV_HPP
#define EAMXX_MAM_WETSCAV_HPP

// For MAM4 aerosol configuration
#include <physics/mam/mam_coupling.hpp>

// For declaring wetscav class derived from atm process class
#include "share/atm_process/atmosphere_process.hpp"

// For MAM4 processes
#include <mam4xx/mam4.hpp>

// For MAM4 calcsize process (FIXME:should we include it in mam4 or mam_coupling??)
#include <mam4xx/modal_aero_calcsize.hpp>

// For wetdep processes
#include <mam4xx/wet_dep.hpp>

// For component name
#include <string>

#include <share/util/scream_common_physics_functions.hpp>
#ifndef KOKKOS_ENABLE_CUDA
#define protected_except_cuda public
#define private_except_cuda public
#else
#define protected_except_cuda protected
#define private_except_cuda private
#endif

namespace scream {

/*
 * The class responsible to handle the aerosol wetscavenging
 *
 * The AD should store exactly ONE instance of this class stored
 * in its list of subcomponents (the AD should make sure of this).
 *
 */

class MAMWetscav : public scream::AtmosphereProcess {

  using KT = ekat::KokkosTypes<DefaultDevice>;
  using view_1d = typename KT::template view_1d<Real>;
  using view_2d = typename KT::template view_2d<Real>;
  using view_3d = typename KT::template view_3d<Real>;

  using const_view_2d = typename KT::template view_2d<const Real>;
  using const_view_1d = typename KT::template view_1d<const Real>; // remove it if possible


  // a thread team dispatched to a single vertical column
  using ThreadTeam = mam4::ThreadTeam;

 public:
  // Constructors
  MAMWetscav(const ekat::Comm &comm, const ekat::ParameterList &params);

  // The type of subcomponent
      AtmosphereProcessType
      type() const override;
  // The name of the subcomponent
  std::string name() const override;

  // Set the grid and input output variables
  void set_grids(
      const std::shared_ptr<const GridsManager> grids_manager) override;

  // management of common atm process memory
  size_t requested_buffer_size_in_bytes() const override;
  void init_buffers(const ATMBufferManager &buffer_manager) override;

  // Initialize variables
  void initialize_impl(const RunType run_type) override;

  // Run the process by one time step
  void run_impl(const double dt) override;

  // Finalize
  void finalize_impl(){/*Do nothing*/};


  // Atmosphere processes often have a pre-processing step that constructs
  // required variables from the set of fields stored in the field manager.
  // This functor implements this step, which is called during run_impl.
  struct Preprocess {
    Preprocess() = default;

    // on host: initializes preprocess functor with necessary state data
    void initialize(const int ncol_in, const int nlev_in,
                    const mam_coupling::WetAtmosphere& wet_atm_in,
                    const mam_coupling::AerosolState& wet_aero_in,
                    const mam_coupling::DryAtmosphere& dry_atm_in,
                    const mam_coupling::AerosolState& dry_aero_in) {
      ncol_pre_     = ncol_in;
      nlev_pre_     = nlev_in;
      wet_atm_pre_  = wet_atm_in;
      wet_aero_pre_ = wet_aero_in;
      dry_atm_pre_  = dry_atm_in;
      dry_aero_pre_ = dry_aero_in;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<KT::ExeSpace>::member_type& team) const {
      const int i = team.league_rank(); // column index
      // first, compute dry fields
      compute_dry_mixing_ratios(team, wet_atm_pre_, dry_atm_pre_, i);
      compute_dry_mixing_ratios(team, wet_atm_pre_, wet_aero_pre_, dry_aero_pre_, i);
      team.team_barrier();
      // second, we can use dry fields to compute dz, zmin, zint
      compute_vertical_layer_heights(team, dry_atm_pre_, i);
      compute_updraft_velocities(team, wet_atm_pre_, dry_atm_pre_, i);
      team.team_barrier();  // allows kernels below to use layer heights
    } // operator()

    // number of horizontal columns and vertical levels
    int ncol_pre_, nlev_pre_;

    // local atmospheric and aerosol state data
    mam_coupling::WetAtmosphere wet_atm_pre_;
    mam_coupling::DryAtmosphere dry_atm_pre_;
    mam_coupling::AerosolState  wet_aero_pre_, dry_aero_pre_;
  }; // MAMWetscav::Preprocess

  // Postprocessing functor
  struct Postprocess {
    Postprocess() = default;

    // on host: initializes postprocess functor with necessary state data
    void initialize(const int ncol, const int nlev,
                    const mam_coupling::WetAtmosphere& wet_atm,
                    const mam_coupling::AerosolState& wet_aero,
                    const mam_coupling::DryAtmosphere& dry_atm,
                    const mam_coupling::AerosolState& dry_aero) {
      ncol_post_ = ncol;
      nlev_post_ = nlev;
      wet_atm_post_ = wet_atm;
      wet_aero_post_ = wet_aero;
      dry_atm_post_ = dry_atm;
      dry_aero_post_ = dry_aero;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<KT::ExeSpace>::member_type& team) const {
      const int i = team.league_rank(); // column index
      compute_wet_mixing_ratios(team, dry_atm_post_, dry_aero_post_, wet_aero_post_, i);
      team.team_barrier();
    } // operator()

    // number of horizontal columns and vertical levels
    int ncol_post_, nlev_post_;

    // local atmospheric and aerosol state data
    mam_coupling::WetAtmosphere wet_atm_post_;
    mam_coupling::DryAtmosphere dry_atm_post_;
    mam_coupling::AerosolState  wet_aero_post_, dry_aero_post_;
  }; // MAMWetscav::Postprocess


  /* -----------------------------------------------
   * Local variables
   * ------------------------------------------------
   */

  // pre- and postprocessing scratch pads (for wet <-> dry conversions)
  Preprocess preprocess_;
  Postprocess postprocess_;


  // Number of horizontal columns and vertical levels
  int ncol_, nlev_;

  //Number of aerosol modes
  static constexpr int ntot_amode_ = mam4::AeroConfig::num_modes();

  // atmospheric variables
  mam_coupling::WetAtmosphere wet_atm_;
  mam_coupling::DryAtmosphere dry_atm_;
  // wet dep
  view_2d work_;
  Kokkos::View<Real * [mam4::aero_model::maxd_aspectype + 2][mam4::aero_model::pcnst]>
        qqcw_sav_;
  // aerosol states
  mam_coupling::AerosolState  wet_aero_, dry_aero_;

  mam_coupling::Buffer buffer_;

  std::shared_ptr<const AbstractGrid> m_grid;
};  // class MAMWetscav

}  // namespace scream

#endif  // EAMXX_MAM_WETSCAV_HPP