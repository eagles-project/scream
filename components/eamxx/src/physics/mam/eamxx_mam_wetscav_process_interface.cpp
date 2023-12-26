#include "physics/mam/eamxx_mam_wetscav_process_interface.hpp"

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
  auto q_unit = kg/kg;
  q_unit.set_string("kg/kg");
  auto n_unit = 1/kg; // units of number mixing ratios of tracers
  n_unit.set_string("#/kg");

  m_grid = grids_manager->get_grid("Physics");
  const auto& grid_name = m_grid->name();

  m_num_cols = m_grid->get_num_local_dofs(); // Number of columns on this rank
  m_num_levs = m_grid->get_num_vertical_levels();  // Number of levels per column

  // Define the different field layouts that will be used for this process
  using namespace ShortFieldTagsNames;

  // Layout for 3D (2d horiz X 1d vertical) variable defined at mid-level and interfaces
  const FieldLayout scalar3d_layout_mid { {COL,LEV}, {m_num_cols,m_num_levs} };

  // -------------------------------------------------------------------------------------------------------------------------
  // These variables are "required" or pure inputs for the process
  // -------------------------------------------------------------------------------------------------------------------------
  add_field<Required>("T_mid",          scalar3d_layout_mid, K,      grid_name);            // temperature [K]
  add_field<Required>("p_mid",          scalar3d_layout_mid, Pa,     grid_name);            // pressure at mid points in [Pa
  add_field<Required>("pseudo_density", scalar3d_layout_mid, Pa,     grid_name);            // pseudo density in [Pa]
  add_field<Required>("qc",             scalar3d_layout_mid, q_unit, grid_name, "tracers"); // liquid cloud water [kg/kg] wet
  add_field<Required>("qi",             scalar3d_layout_mid, q_unit, grid_name, "tracers"); // ice cloud water [kg/kg] wet

  // -- Input variables that exists in PBUF in EAM
  static constexpr auto nondim = Units::nondimensional();
  add_field<Required>("cldn",          scalar3d_layout_mid, nondim,  grid_name);  // layer cloud fraction [fraction]
  add_field<Required>("rprdsh",        scalar3d_layout_mid, kg/kg/s, grid_name); // rain production, shallow convection [kg/kg/s]
  add_field<Required>("rprddp",        scalar3d_layout_mid, kg/kg/s, grid_name); // rain production, deep convection [kg/kg/s]
  add_field<Required>("evapcsh",       scalar3d_layout_mid, kg/kg/s, grid_name); // Evaporation rate of shallow convective precipitation >=0. [kg/kg/s]
  add_field<Required>("evapcdp",       scalar3d_layout_mid, kg/kg/s, grid_name); // Evaporation rate of deep convective precipitation >=0. [kg/kg/s]

  // -- Input variables that exists in PBUF in EAM (in wetdep.F90) in the "inputs" data structure
  add_field<Required>("cldt",          scalar3d_layout_mid, nondim,  grid_name); // total cloud fraction [fraction]
  add_field<Required>("qme",           scalar3d_layout_mid, kg/kg/s, grid_name); // net condensation/evaporation of cloud water [kg/kg/s]
  add_field<Required>("prain",         scalar3d_layout_mid, kg/kg/s, grid_name); // stratiform rain production rate [kg/kg/s]
  add_field<Required>("evapr",         scalar3d_layout_mid, kg/kg/s, grid_name); // evaporation from stratiform rain [kg/kg/s]

  // -- Input variables that exists in PBUF in EAM (in wetdep.F90)
  add_field<Required>("icwmrdp",       scalar3d_layout_mid, kg/kg,   grid_name); // In cloud water mixing ratio, deep convection [kg/kg]
  add_field<Required>("icwmrsh",       scalar3d_layout_mid, kg/kg,   grid_name); // In cloud water mixing ratio, shallow convection [kg/kg]
  add_field<Required>("rprddp",        scalar3d_layout_mid, kg/kg/s, grid_name); // Rain production, deep convection [kg/kg/s]
  add_field<Required>("rprdsh",        scalar3d_layout_mid, kg/kg/s, grid_name); // Rain production, shallow convection [kg/kg/s]
  add_field<Required>("sh_frac",       scalar3d_layout_mid, nondim,  grid_name); // Shallow convective cloud fraction [fraction]
  add_field<Required>("dp_frac",       scalar3d_layout_mid, nondim,  grid_name); // Deep convective cloud fraction [fraction]
  add_field<Required>("evapcsh",       scalar3d_layout_mid, kg/kg/s, grid_name); // Evaporation rate of shallow convective precipitation >=0. [kg/kg/s]
  add_field<Required>("evapcdp",       scalar3d_layout_mid, kg/kg/s, grid_name); // Evaporation rate of deep convective precipitation >=0. [kg/kg/s]

  // -------------------------------------------------------------------------------------------------------------------------
  // These variables are "updated" or inputs/outputs for the process
  // -------------------------------------------------------------------------------------------------------------------------
  
  // -- surface fluxes (input/outpts) for the coupler's cam_out data struture for the land model
  static constexpr auto m2 = m*m;
  add_field<Updated>("bcphiwet",      scalar3d_layout_mid, kg/m2/s, grid_name); // wet deposition of hydrophilic black carbon [kg/m2/s]
  add_field<Updated>("bcphidry",      scalar3d_layout_mid, kg/m2/s, grid_name); // dry deposition of hydrophilic black carbon [kg/m2/s]
  add_field<Updated>("ocphiwet",      scalar3d_layout_mid, kg/m2/s, grid_name); // wet deposition of hydrophilic organic carbon [kg/m2/s]
  add_field<Updated>("ocphidry",      scalar3d_layout_mid, kg/m2/s, grid_name); // dry deposition of hydrophilic organic carbon [kg/m2/s]

  add_field<Updated>("dstwet1",       scalar3d_layout_mid, kg/m2/s, grid_name); // wet deposition of dust (bin1) [kg/m2/s]
  add_field<Updated>("dstwet2",       scalar3d_layout_mid, kg/m2/s, grid_name); // wet deposition of dust (bin2) [kg/m2/s]
  add_field<Updated>("dstwet3",       scalar3d_layout_mid, kg/m2/s, grid_name); // wet deposition of dust (bin3) [kg/m2/s]
  add_field<Updated>("dstwet4",       scalar3d_layout_mid, kg/m2/s, grid_name); // wet deposition of dust (bin4) [kg/m2/s]

  // -- input/ouputs from PBUF for updating particle size and water uptake by particles  
  static constexpr auto m3 = m2*m;
  add_field<Updated>("dgncur_a",      scalar3d_layout_mid, m,       grid_name); // aerosol particle diameter [m]
  add_field<Updated>("wetdens",       scalar3d_layout_mid, kg/m3,   grid_name); // wet aerosol density [kg/m3]
  add_field<Updated>("qaerwat",       scalar3d_layout_mid, kg/kg,   grid_name); // aerosol water [kg/kg]
  add_field<Updated>("dgnumwet",      scalar3d_layout_mid, m,       grid_name); // wet aerosol diameter [m]
  add_field<Updated>("fracis",        scalar3d_layout_mid, nondim,  grid_name); // fraction of transported species that are insoluble [fraction]

  // -- interstitial and cloudborne aerosol tracers of interest: mass (q) and number (n) mixing ratios
  // -- NOTE: Interstitial aerosols are updated in the interface using the "tendencies" from the wetscavenging process
  for (int m = 0; m < mam_coupling::num_aero_modes(); ++m) {
    //interstitial aerosol tracers of interest: number (n) mixing ratios
    const char* int_nmr_field_name = mam_coupling::int_aero_nmr_field_name(m);
    add_field<Updated>(int_nmr_field_name, scalar3d_layout_mid, n_unit, grid_name, "tracers");

    //cloudborne aerosol tracers of interest: number (n) mixing ratios
    const char* cld_nmr_field_name = mam_coupling::cld_aero_nmr_field_name(m);

    //NOTE: DO NOT add cld borne aerosols to the "tracer" group as these are NOT advected
    add_field<Updated>(cld_nmr_field_name, scalar3d_layout_mid, n_unit, grid_name); 

    for (int a = 0; a < mam_coupling::num_aero_species(); ++a) {
      // (interstitial) aerosol tracers of interest: mass (q) mixing ratios
      const char* int_mmr_field_name = mam_coupling::int_aero_mmr_field_name(m, a);
      if (strlen(int_mmr_field_name) > 0) {
        add_field<Updated>(int_mmr_field_name, scalar3d_layout_mid, q_unit, grid_name, "tracers");
      }
    
      // (cloudborne) aerosol tracers of interest: mass (q) mixing ratios
      const char* cld_mmr_field_name = mam_coupling::cld_aero_mmr_field_name(m, a);
      if (strlen(cld_mmr_field_name) > 0) {
        //NOTE: DO NOT add cld borne aerosols to the "tracer" group as these are NOT advected
        add_field<Updated>(cld_mmr_field_name, scalar3d_layout_mid, q_unit, grid_name);
      }
    }
  }
}

// =========================================================================================
void MAMWetscav::initialize_impl (const RunType run_type)
{
  // Gather runtime options
  //(e.g.) runtime_options.lambda_low    = m_params.get<double>("lambda_low");

  
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
