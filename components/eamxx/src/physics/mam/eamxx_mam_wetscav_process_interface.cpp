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
void MAMWetscav::
set_computed_group_impl (const FieldGroup& group)
{
  EKAT_REQUIRE_MSG(group.m_info->size() >= 3,
                   "Error! Shoc requires at least 3 tracers (tke, qv, qc) as inputs.");

  const auto& name = group.m_info->m_group_name;

  EKAT_REQUIRE_MSG(name=="tracers",
    "Error! We were not expecting a field group called '" << name << "\n");

  EKAT_REQUIRE_MSG(group.m_info->m_bundled,
      "Error! Shoc expects bundled fields for tracers.\n");

  // Calculate number of advected tracers
  m_num_tracers = group.m_info->size();
}

// =========================================================================================
size_t MAMWetscav::requested_buffer_size_in_bytes() const
{
  const int nlev_packs       = ekat::npack<Spack>(m_num_levs);
  const int nlevi_packs      = ekat::npack<Spack>(m_num_levs+1);
  const int num_tracer_packs = ekat::npack<Spack>(m_num_tracers);

  // Number of Reals needed by local views in the interface
  const size_t interface_request = Buffer::num_1d_scalar_ncol*m_num_cols*sizeof(Real) +
                                   Buffer::num_1d_scalar_nlev*nlev_packs*sizeof(Spack) +
                                   Buffer::num_2d_vector_mid*m_num_cols*nlev_packs*sizeof(Spack) +
                                   Buffer::num_2d_vector_int*m_num_cols*nlevi_packs*sizeof(Spack) +
                                   Buffer::num_2d_vector_tr*m_num_cols*num_tracer_packs*sizeof(Spack);

  // Number of Reals needed by the WorkspaceManager passed to shoc_main
  const auto policy       = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, nlev_packs);
  const int n_wind_slots  = ekat::npack<Spack>(2)*Spack::n;
  const int n_trac_slots  = ekat::npack<Spack>(m_num_tracers+3)*Spack::n;
  const size_t wsm_request= WSM::get_total_bytes_needed(nlevi_packs, 14+(n_wind_slots+n_trac_slots), policy);

  return interface_request + wsm_request;
}

// =========================================================================================
void MAMWetscav::init_buffers(const ATMBufferManager &buffer_manager)
{
  EKAT_REQUIRE_MSG(buffer_manager.allocated_bytes() >= requested_buffer_size_in_bytes(), "Error! Buffers size not sufficient.\n");

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
  EKAT_REQUIRE_MSG(used_mem==requested_buffer_size_in_bytes(), "Error! Used memory != requested memory for MAMWetscav.");
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
void MAMWetscav::apply_turbulent_mountain_stress()
{
  auto surf_drag_coeff_tms = get_field_in("surf_drag_coeff_tms").get_view<const Real*>();
  auto horiz_winds         = get_field_in("horiz_winds").get_view<const Spack***>();

  auto rrho_i   = m_buffer.rrho_i;
  auto upwp_sfc = m_buffer.upwp_sfc;
  auto vpwp_sfc = m_buffer.vpwp_sfc;

  const int nlev_v  = (m_num_levs-1)/Spack::n;
  const int nlev_p  = (m_num_levs-1)%Spack::n;
  const int nlevi_v = m_num_levs/Spack::n;
  const int nlevi_p = m_num_levs%Spack::n;

  Kokkos::parallel_for("apply_tms", KT::RangePolicy(0, m_num_cols), KOKKOS_LAMBDA (const int i) {
    upwp_sfc(i) -= surf_drag_coeff_tms(i)*horiz_winds(i,0,nlev_v)[nlev_p]/rrho_i(i,nlevi_v)[nlevi_p];
    vpwp_sfc(i) -= surf_drag_coeff_tms(i)*horiz_winds(i,1,nlev_v)[nlev_p]/rrho_i(i,nlevi_v)[nlevi_p];
  });
}
// =========================================================================================
void MAMWetscav::check_flux_state_consistency(const double dt)
{
  using PC = scream::physics::Constants<Real>;
  const Real gravit = PC::gravit;
  const Real qmin   = 1e-12; // minimum permitted constituent concentration (kg/kg)

  const auto& pseudo_density = get_field_in ("pseudo_density").get_view<const Spack**>();
  const auto& surf_evap      = get_field_out("surf_evap").get_view<Real*>();
  const auto& qv             = get_field_out("qv").get_view<Spack**>();

  const auto nlevs           = m_num_levs;
  const auto nlev_packs      = ekat::npack<Spack>(nlevs);
  const auto last_pack_idx   = (nlevs-1)/Spack::n;
  const auto last_pack_entry = (nlevs-1)%Spack::n;
  const auto policy          = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, nlev_packs);
  Kokkos::parallel_for("check_flux_state_consistency",
                       policy,
                       KOKKOS_LAMBDA (const KT::MemberType& team) {
    const auto i = team.league_rank();

    const auto& pseudo_density_i = ekat::subview(pseudo_density, i);
    const auto& qv_i             = ekat::subview(qv, i);

    // reciprocal of pseudo_density at the bottom layer
    const auto rpdel = 1.0/pseudo_density_i(last_pack_idx)[last_pack_entry];

    // Check if the negative surface latent heat flux can exhaust
    // the moisture in the lowest model level. If so, apply fixer.
    const auto condition = surf_evap(i) - (qmin - qv_i(last_pack_idx)[last_pack_entry])/(dt*gravit*rpdel);
    if (condition < 0) {
      const auto cc = abs(surf_evap(i)*dt*gravit);

      auto tracer_mass = [&](const int k) {
        return qv_i(k)*pseudo_density_i(k);
      };
      Real mm = ekat::ExeSpaceUtils<KT::ExeSpace>::view_reduction(team, 0, nlevs, tracer_mass);

      EKAT_KERNEL_ASSERT_MSG(mm >= cc, "Error! Total mass of column vapor should be greater than mass of surf_evap.\n");

      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev_packs), [&](const int& k) {
        const auto adjust = cc*qv_i(k)*pseudo_density_i(k)/mm;
        qv_i(k) = (qv_i(k)*pseudo_density_i(k) - adjust)/pseudo_density_i(k);
      });

      surf_evap(i) = 0;
    }
  });
}
// =========================================================================================
} // namespace scream
