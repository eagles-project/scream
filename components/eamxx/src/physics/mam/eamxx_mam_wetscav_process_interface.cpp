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
   * Like universal constants, shoc options.
   */
}

// =========================================================================================
void MAMWetscav::set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{
  using namespace ekat::units;

  // The units of mixing ratio Q are technically non-dimensional.
  // Nevertheless, for output reasons, we like to see 'kg/kg'.
  auto Qunit = kg/kg;
  Qunit.set_string("kg/kg");
  auto nondim = Units::nondimensional();

  m_grid = grids_manager->get_grid("Physics");
  const auto& grid_name = m_grid->name();

  m_num_cols = m_grid->get_num_local_dofs(); // Number of columns on this rank
  m_num_levs = m_grid->get_num_vertical_levels();  // Number of levels per column

  m_cell_area = m_grid->get_geometry_data("area").get_view<const Real*>(); // area of each cell
  m_cell_lat  = m_grid->get_geometry_data("lat").get_view<const Real*>(); // area of each cell

  // Define the different field layouts that will be used for this process
  using namespace ShortFieldTagsNames;

  // Layout for 2D (1d horiz X 1d vertical) variable
  FieldLayout scalar2d_layout_col{ {COL}, {m_num_cols} };

  // Layout for surf_mom_flux
  FieldLayout surf_mom_flux_layout { {COL, CMP}, {m_num_cols, 2} };

  // Layout for 3D (2d horiz X 1d vertical) variable defined at mid-level and interfaces
  FieldLayout scalar3d_layout_mid { {COL,LEV}, {m_num_cols,m_num_levs} };
  FieldLayout scalar3d_layout_int { {COL,ILEV}, {m_num_cols,m_num_levs+1} };

  // Layout for horiz_wind field
  FieldLayout horiz_wind_layout { {COL,CMP,LEV}, {m_num_cols,2,m_num_levs} };

  // Define fields needed in SHOC.
  // Note: shoc_main is organized by a set of 5 structures, variables below are organized
  //       using the same approach to make it easier to follow.

  constexpr int ps = Spack::n;

  const auto m2 = m*m;
  const auto s2 = s*s;

  // These variables are needed by the interface, but not actually passed to shoc_main.
  add_field<Required>("omega",               scalar3d_layout_mid,  Pa/s,    grid_name, ps);
  add_field<Required>("surf_sens_flux",      scalar2d_layout_col,  W/m2,    grid_name);
  add_field<Required>("surf_mom_flux",       surf_mom_flux_layout, N/m2, grid_name);

  add_field<Updated>("surf_evap",           scalar2d_layout_col,  kg/m2/s, grid_name);
  add_field<Updated> ("T_mid",               scalar3d_layout_mid,  K,       grid_name, ps);
  add_field<Updated> ("qv",                  scalar3d_layout_mid,  Qunit,   grid_name, "tracers", ps);

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
