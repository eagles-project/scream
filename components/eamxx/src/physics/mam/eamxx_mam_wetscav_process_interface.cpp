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

  // If TMS is a process, add surface drag coefficient to required fields
  if (m_params.get<bool>("apply_tms", false)) {
    add_field<Required>("surf_drag_coeff_tms", scalar2d_layout_col,  kg/s/m2, grid_name);
  }

  // Input variables
  add_field<Required>("p_mid",          scalar3d_layout_mid, Pa,    grid_name, ps);
  add_field<Required>("p_int",          scalar3d_layout_int, Pa,    grid_name, ps);
  add_field<Required>("pseudo_density", scalar3d_layout_mid, Pa,    grid_name, ps);
  add_field<Required>("phis",           scalar2d_layout_col, m2/s2, grid_name, ps);

  // Input/Output variables
  add_field<Updated>("tke",           scalar3d_layout_mid, m2/s2,   grid_name, "tracers", ps);
  add_field<Updated>("horiz_winds",   horiz_wind_layout,   m/s,     grid_name, ps);
  add_field<Updated>("sgs_buoy_flux", scalar3d_layout_mid, K*(m/s), grid_name, ps);
  add_field<Updated>("eddy_diff_mom", scalar3d_layout_mid, m2/s,    grid_name, ps);
  add_field<Updated>("qc",            scalar3d_layout_mid, Qunit,   grid_name, "tracers", ps);
  add_field<Updated>("cldfrac_liq",   scalar3d_layout_mid, nondim,  grid_name, ps);

  // Output variables
  add_field<Computed>("pbl_height",    scalar2d_layout_col, m,           grid_name);
  add_field<Computed>("inv_qc_relvar", scalar3d_layout_mid, Qunit*Qunit, grid_name, ps);

  // Tracer group
  add_group<Updated>("tracers", grid_name, ps, Bundling::Required);

  // Boundary flux fields for energy and mass conservation checks
  if (has_column_conservation_check()) {
    add_field<Computed>("vapor_flux", scalar2d_layout_col, kg/m2/s, grid_name);
    add_field<Computed>("water_flux", scalar2d_layout_col, m/s,     grid_name);
    add_field<Computed>("ice_flux",   scalar2d_layout_col, m/s,     grid_name);
    add_field<Computed>("heat_flux",  scalar2d_layout_col, W/m2,    grid_name);
  }
}



// =========================================================================================
void MAMWetscav::init_buffers(const ATMBufferManager &buffer_manager)
{
  Real* mem = reinterpret_cast<Real*>(buffer_manager.get_memory());

  // 1d scalar views
  using scalar_view_t = decltype(m_buffer.cell_length);
  scalar_view_t* _1d_scalar_view_ptrs[Buffer::num_1d_scalar_ncol] =
    {&m_buffer.cell_length, &m_buffer.wpthlp_sfc, &m_buffer.wprtp_sfc, &m_buffer.upwp_sfc, &m_buffer.vpwp_sfc
#ifdef SCREAM_SMALL_KERNELS
     , &m_buffer.se_b, &m_buffer.ke_b, &m_buffer.wv_b, &m_buffer.wl_b
     , &m_buffer.se_a, &m_buffer.ke_a, &m_buffer.wv_a, &m_buffer.wl_a
     , &m_buffer.ustar, &m_buffer.kbfs, &m_buffer.obklen, &m_buffer.ustar2, &m_buffer.wstar
#endif
    };
  for (int i = 0; i < Buffer::num_1d_scalar_ncol; ++i) {
    *_1d_scalar_view_ptrs[i] = scalar_view_t(mem, m_num_cols);
    mem += _1d_scalar_view_ptrs[i]->size();
  }

  Spack* s_mem = reinterpret_cast<Spack*>(mem);

  // 2d packed views
  const int nlev_packs       = ekat::npack<Spack>(m_num_levs);
  const int nlevi_packs      = ekat::npack<Spack>(m_num_levs+1);
  const int num_tracer_packs = ekat::npack<Spack>(m_num_tracers);

  m_buffer.pref_mid = decltype(m_buffer.pref_mid)(s_mem, nlev_packs);
  s_mem += m_buffer.pref_mid.size();

  using spack_2d_view_t = decltype(m_buffer.z_mid);
  spack_2d_view_t* _2d_spack_mid_view_ptrs[Buffer::num_2d_vector_mid] = {
    &m_buffer.z_mid, &m_buffer.rrho, &m_buffer.thv, &m_buffer.dz, &m_buffer.zt_grid, &m_buffer.wm_zt,
    &m_buffer.inv_exner, &m_buffer.thlm, &m_buffer.qw, &m_buffer.dse, &m_buffer.tke_copy, &m_buffer.qc_copy,
    &m_buffer.shoc_ql2, &m_buffer.shoc_mix, &m_buffer.isotropy, &m_buffer.w_sec, &m_buffer.wqls_sec, &m_buffer.brunt
#ifdef SCREAM_SMALL_KERNELS
    , &m_buffer.rho_zt, &m_buffer.shoc_qv, &m_buffer.tabs, &m_buffer.dz_zt, &m_buffer.tkh
#endif
  };

  spack_2d_view_t* _2d_spack_int_view_ptrs[Buffer::num_2d_vector_int] = {
    &m_buffer.z_int, &m_buffer.rrho_i, &m_buffer.zi_grid, &m_buffer.thl_sec, &m_buffer.qw_sec,
    &m_buffer.qwthl_sec, &m_buffer.wthl_sec, &m_buffer.wqw_sec, &m_buffer.wtke_sec, &m_buffer.uw_sec,
    &m_buffer.vw_sec, &m_buffer.w3
#ifdef SCREAM_SMALL_KERNELS
    , &m_buffer.dz_zi
#endif
  };

  for (int i = 0; i < Buffer::num_2d_vector_mid; ++i) {
    *_2d_spack_mid_view_ptrs[i] = spack_2d_view_t(s_mem, m_num_cols, nlev_packs);
    s_mem += _2d_spack_mid_view_ptrs[i]->size();
  }

  for (int i = 0; i < Buffer::num_2d_vector_int; ++i) {
    *_2d_spack_int_view_ptrs[i] = spack_2d_view_t(s_mem, m_num_cols, nlevi_packs);
    s_mem += _2d_spack_int_view_ptrs[i]->size();
  }
  m_buffer.wtracer_sfc = decltype(m_buffer.wtracer_sfc)(s_mem, m_num_cols, num_tracer_packs);
  s_mem += m_buffer.wtracer_sfc.size();

  // WSM data
  m_buffer.wsm_data = s_mem;

  // Compute workspace manager size to check used memory
  // vs. requested memory
  const auto policy      = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, nlev_packs);
  const int n_wind_slots = ekat::npack<Spack>(2)*Spack::n;
  const int n_trac_slots = ekat::npack<Spack>(m_num_tracers+3)*Spack::n;
  const int wsm_size     = WSM::get_total_bytes_needed(nlevi_packs, 14+(n_wind_slots+n_trac_slots), policy)/sizeof(Spack);
  s_mem += wsm_size;

  size_t used_mem = (reinterpret_cast<Real*>(s_mem) - buffer_manager.get_memory())*sizeof(Real);
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
