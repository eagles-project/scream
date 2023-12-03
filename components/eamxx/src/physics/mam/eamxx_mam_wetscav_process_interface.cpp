#include "ekat/ekat_assert.hpp"
#include "physics/mam/eamxx_mam_wetscav_process_interface.hpp"

#include "share/property_checks/field_lower_bound_check.hpp"
#include "share/property_checks/field_within_interval_check.hpp"

#include "scream_config.h" // for SCREAM_CIME_BUILD

namespace scream
{

// =========================================================================================
MAMWetscav::MAMWetscav (const ekat::Comm& comm,const ekat::ParameterList& params)
  : AtmosphereProcess(comm, params)
{
  /* Anything that can be initialized without grid information can be initialized here.
   * Like universal constants, mam wetscav options.
   */
}

// =========================================================================================
void MAMWetscav::set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{
  using namespace ekat::units;

  // The units of mixing ratio Q are technically non-dimensional.
  // Nevertheless, for output reasons, we like to see 'kg/kg'.
  auto Qunit = kg/kg;

  m_grid = grids_manager->get_grid("Physics");
  const auto& grid_name = m_grid->name();

  m_num_cols = m_grid->get_num_local_dofs(); // Number of columns on this rank
  m_num_levs = m_grid->get_num_vertical_levels();  // Number of levels per column


  // Define the different field layouts that will be used for this process
  using namespace ShortFieldTagsNames;


  // Layout for 3D (2d horiz X 1d vertical) variable defined at mid-level and interfaces
  FieldLayout scalar3d_layout_mid { {COL,LEV}, {m_num_cols,m_num_levs} };





  // These variables are needed by the interface.
  add_field<Updated> ("T_mid",               scalar3d_layout_mid,  K,       grid_name);
  
}

// =========================================================================================
void MAMWetscav::initialize_impl (const RunType run_type)
{
  
}

// =========================================================================================
void MAMWetscav::run_impl (const double dt)
{

}
// =========================================================================================
void MAMWetscav::finalize_impl()
{
  // Do nothing
}

// =========================================================================================
} // namespace scream
