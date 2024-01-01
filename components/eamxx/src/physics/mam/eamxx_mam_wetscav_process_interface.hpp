#ifndef EAMXX_MAM_WETSCAV_HPP
#define EAMXX_MAM_WETSCAV_HPP

// For MAM4 aerosol configuration
#include <physics/mam/mam_coupling.hpp>

// For declaring wetscav class derived from atm process class
#include "share/atm_process/atmosphere_process.hpp"

// For MAM4 processes
#include <mam4xx/mam4.hpp>

// For component name
#include <string>

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

  // Initialize variables
  void initialize_impl(const RunType run_type) override;

  // Run the process by one time step
  void run_impl(const double dt) override;

  // Finalize
  void finalize_impl(){/*Do nothing*/};

  /* -----------------------------------------------
   * Local variables
   * ------------------------------------------------
   */
  // Number of horizontal columns and vertical levels
  int ncol_, nlev_;

  // MAM configuration (particle number and size description)
  mam4::AeroConfig aero_config_;


  // atmospheric variables
  mam_coupling::WetAtmosphere wet_atm_;
  mam_coupling::DryAtmosphere dry_atm_;

  mam4::CalcSize calcsize_;

  //const_view_1d dummy_("DummyView", nlev);


  std::shared_ptr<const AbstractGrid> m_grid;
};  // class MAMWetscav

}  // namespace scream

#endif  // EAMXX_MAM_WETSCAV_HPP
