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

  using const_view_2d = typename KT::template view_2d<const Real>;


  // a thread team dispatched to a single vertical column
  using ThreadTeam = mam4::ThreadTeam;

 public:
  // Constructors
  MAMWetscav(const ekat::Comm &comm, const ekat::ParameterList &params);

  // The type of subcomponent
  AtmosphereProcessType type() const { return AtmosphereProcessType::Physics; }

  // The name of the subcomponent
  std::string name() const { return "mam_wetscavenging"; }

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

      compute_dry_mixing_ratios(team, wet_atm_pre_, dry_atm_pre_, i);
      compute_vertical_layer_heights(team, dry_atm_pre_, i);
      team.team_barrier(); // allows kernels below to use layer heights
      compute_updraft_velocities(team, wet_atm_pre_, dry_atm_pre_, i);
      team.team_barrier();
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

  //Extent for the e3sm's state vector for tracers
  //--NOTE: The aerosol species are from index 16 to 40 ( or 15 to 39 in C++)
  //        but we define this variable from 0 to nvars_, where nvars_ is 39.
  //        Index 0 to 14 has no value
  static constexpr int nvars_ = mam4::ndrop::nvars;


  // atmospheric variables
  mam_coupling::WetAtmosphere wet_atm_;
  mam_coupling::DryAtmosphere dry_atm_;

  const_view_2d cldn_prev_step_;//, cldt_prev_step_; // cloud fraction from the previous step, FIXME: they carry same info, we might remove one later

  // aerosol states
  mam_coupling::AerosolState  wet_aero_, dry_aero_;

  // workspace manager for internal local variables
  //ekat::WorkspaceManager<Real, KT::Device> workspace_mgr_;
  mam_coupling::Buffer buffer_;

  mam4::WetDeposition wetdep_;

  
                                         
  std::shared_ptr<const AbstractGrid> m_grid;
};  // class MAMWetscav

}  // namespace scream

#endif  // EAMXX_MAM_WETSCAV_HPP