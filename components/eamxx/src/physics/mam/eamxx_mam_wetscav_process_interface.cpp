#include "physics/mam/eamxx_mam_wetscav_process_interface.hpp"

namespace scream {

// =========================================================================================
MAMWetscav::MAMWetscav(const ekat::Comm &comm,
                       const ekat::ParameterList &params)
    : AtmosphereProcess(comm, params) {
  /* Anything that can be initialized without grid information can be
   * initialized here. Like universal constants, mam wetscav options.
   */
}

// =========================================================================================
void MAMWetscav::set_grids(
    const std::shared_ptr<const GridsManager> grids_manager) {
  using namespace ekat::units;

  // The units of mixing ratio Q are technically non-dimensional.
  // Nevertheless, for output reasons, we like to see 'kg/kg'.
  auto q_unit = kg / kg;
  q_unit.set_string("kg/kg");
  auto n_unit = 1 / kg;  // units of number mixing ratios of tracers
  n_unit.set_string("#/kg");

  m_grid                = grids_manager->get_grid("Physics");
  const auto &grid_name = m_grid->name();

  ncol_ = m_grid->get_num_local_dofs();       // Number of columns on this rank
  nlev_ = m_grid->get_num_vertical_levels();  // Number of levels per column

  // Define the different field layouts that will be used for this process
  using namespace ShortFieldTagsNames;

  // Layout for 3D (2d horiz X 1d vertical) variable defined at mid-level and
  // interfaces
  const FieldLayout scalar3d_layout_mid{{COL, LEV}, {ncol_, nlev_}};

  // -------------------------------------------------------------------------------------------------------------------------
  // These variables are "required" or pure inputs for the process
  // -------------------------------------------------------------------------------------------------------------------------
  add_field<Required>("T_mid", scalar3d_layout_mid, K,
                      grid_name);  // temperature [K]
  add_field<Required>("p_mid", scalar3d_layout_mid, Pa,
                      grid_name);  // pressure at mid points in [Pa
  add_field<Required>("pseudo_density", scalar3d_layout_mid, Pa,
                      grid_name);  // pseudo density in [Pa]
  add_field<Required>("qc", scalar3d_layout_mid, q_unit, grid_name,
                      "tracers");  // liquid cloud water [kg/kg] wet
  add_field<Required>("qi", scalar3d_layout_mid, q_unit, grid_name,
                      "tracers");  // ice cloud water [kg/kg] wet

  // -- Input variables that exists in PBUF in EAM
  static constexpr auto nondim = Units::nondimensional();
  add_field<Required>("cldn", scalar3d_layout_mid, nondim,
                      grid_name);  // layer cloud fraction [fraction]
  add_field<Required>(
      "rprdsh", scalar3d_layout_mid, kg / kg / s,
      grid_name);  // rain production, shallow convection [kg/kg/s]
  add_field<Required>("rprddp", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // rain production, deep convection [kg/kg/s]
  add_field<Required>("evapcsh", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // Evaporation rate of shallow convective
                                   // precipitation >=0. [kg/kg/s]
  add_field<Required>("evapcdp", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // Evaporation rate of deep convective
                                   // precipitation >=0. [kg/kg/s]

  // -- Input variables that exists in PBUF in EAM (in wetdep.F90) in the
  // "inputs" data structure
  add_field<Required>("cldt", scalar3d_layout_mid, nondim,
                      grid_name);  // total cloud fraction [fraction]
  add_field<Required>(
      "qme", scalar3d_layout_mid, kg / kg / s,
      grid_name);  // net condensation/evaporation of cloud water [kg/kg/s]
  add_field<Required>("prain", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // stratiform rain production rate [kg/kg/s]
  add_field<Required>("evapr", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // evaporation from stratiform rain [kg/kg/s]

  // -- Input variables that exists in PBUF in EAM (in wetdep.F90)
  add_field<Required>(
      "icwmrdp", scalar3d_layout_mid, kg / kg,
      grid_name);  // In cloud water mixing ratio, deep convection [kg/kg]
  add_field<Required>(
      "icwmrsh", scalar3d_layout_mid, kg / kg,
      grid_name);  // In cloud water mixing ratio, shallow convection [kg/kg]
  add_field<Required>("rprddp", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // Rain production, deep convection [kg/kg/s]
  add_field<Required>(
      "rprdsh", scalar3d_layout_mid, kg / kg / s,
      grid_name);  // Rain production, shallow convection [kg/kg/s]
  add_field<Required>(
      "sh_frac", scalar3d_layout_mid, nondim,
      grid_name);  // Shallow convective cloud fraction [fraction]
  add_field<Required>("dp_frac", scalar3d_layout_mid, nondim,
                      grid_name);  // Deep convective cloud fraction [fraction]
  add_field<Required>("evapcsh", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // Evaporation rate of shallow convective
                                   // precipitation >=0. [kg/kg/s]
  add_field<Required>("evapcdp", scalar3d_layout_mid, kg / kg / s,
                      grid_name);  // Evaporation rate of deep convective
                                   // precipitation >=0. [kg/kg/s]

  // -------------------------------------------------------------------------------------------------------------------------
  // These variables are "updated" or inputs/outputs for the process
  // -------------------------------------------------------------------------------------------------------------------------

  // -- surface fluxes (input/outpts) for the coupler's cam_out data struture
  // for the land model
  static constexpr auto m2 = m * m;
  add_field<Updated>(
      "bcphiwet", scalar3d_layout_mid, kg / m2 / s,
      grid_name);  // wet deposition of hydrophilic black carbon [kg/m2/s]
  add_field<Updated>(
      "bcphidry", scalar3d_layout_mid, kg / m2 / s,
      grid_name);  // dry deposition of hydrophilic black carbon [kg/m2/s]
  add_field<Updated>(
      "ocphiwet", scalar3d_layout_mid, kg / m2 / s,
      grid_name);  // wet deposition of hydrophilic organic carbon [kg/m2/s]
  add_field<Updated>(
      "ocphidry", scalar3d_layout_mid, kg / m2 / s,
      grid_name);  // dry deposition of hydrophilic organic carbon [kg/m2/s]

  add_field<Updated>("dstwet1", scalar3d_layout_mid, kg / m2 / s,
                     grid_name);  // wet deposition of dust (bin1) [kg/m2/s]
  add_field<Updated>("dstwet2", scalar3d_layout_mid, kg / m2 / s,
                     grid_name);  // wet deposition of dust (bin2) [kg/m2/s]
  add_field<Updated>("dstwet3", scalar3d_layout_mid, kg / m2 / s,
                     grid_name);  // wet deposition of dust (bin3) [kg/m2/s]
  add_field<Updated>("dstwet4", scalar3d_layout_mid, kg / m2 / s,
                     grid_name);  // wet deposition of dust (bin4) [kg/m2/s]

  // -- input/ouputs from PBUF for updating particle size and water uptake by
  // particles
  static constexpr auto m3 = m2 * m;
  add_field<Updated>("dgncur_a", scalar3d_layout_mid, m,
                     grid_name);  // aerosol particle diameter [m]
  add_field<Updated>("wetdens", scalar3d_layout_mid, kg / m3,
                     grid_name);  // wet aerosol density [kg/m3]
  add_field<Updated>("qaerwat", scalar3d_layout_mid, kg / kg,
                     grid_name);  // aerosol water [kg/kg]
  add_field<Updated>("dgnumwet", scalar3d_layout_mid, m,
                     grid_name);  // wet aerosol diameter [m]
  add_field<Updated>("fracis", scalar3d_layout_mid, nondim,
                     grid_name);  // fraction of transported species that are
                                  // insoluble [fraction]

  // -- interstitial and cloudborne aerosol tracers of interest: mass (q) and
  // number (n) mixing ratios
  // -- NOTE: Interstitial aerosols are updated in the interface using the
  // "tendencies" from the wetscavenging process
  for(int m = 0; m < mam_coupling::num_aero_modes(); ++m) {
    // interstitial aerosol tracers of interest: number (n) mixing ratios
    const char *int_nmr_field_name = mam_coupling::int_aero_nmr_field_name(m);
    add_field<Updated>(int_nmr_field_name, scalar3d_layout_mid, n_unit,
                       grid_name, "tracers");

    // cloudborne aerosol tracers of interest: number (n) mixing ratios
    const char *cld_nmr_field_name = mam_coupling::cld_aero_nmr_field_name(m);

    // NOTE: DO NOT add cld borne aerosols to the "tracer" group as these are
    // NOT advected
    add_field<Updated>(cld_nmr_field_name, scalar3d_layout_mid, n_unit,
                       grid_name);

    for(int a = 0; a < mam_coupling::num_aero_species(); ++a) {
      // (interstitial) aerosol tracers of interest: mass (q) mixing ratios
      const char *int_mmr_field_name =
          mam_coupling::int_aero_mmr_field_name(m, a);
      if(strlen(int_mmr_field_name) > 0) {
        add_field<Updated>(int_mmr_field_name, scalar3d_layout_mid, q_unit,
                           grid_name, "tracers");
      }

      // (cloudborne) aerosol tracers of interest: mass (q) mixing ratios
      const char *cld_mmr_field_name =
          mam_coupling::cld_aero_mmr_field_name(m, a);
      if(strlen(cld_mmr_field_name) > 0) {
        // NOTE: DO NOT add cld borne aerosols to the "tracer" group as these
        // are NOT advected
        add_field<Updated>(cld_mmr_field_name, scalar3d_layout_mid, q_unit,
                           grid_name);
      }
    }
  }
}

// =========================================================================================
void MAMWetscav::initialize_impl(const RunType run_type) {
  // Gather runtime options
  //(e.g.) runtime_options.lambda_low    = m_params.get<double>("lambda_low");

  // populate the wet and dry atmosphere states with views from fields and
  // the buffer
  wet_atm_.qc = get_field_in("qc").get_view<const Real **>();
  wet_atm_.qi = get_field_in("qi").get_view<const Real **>();

  dry_atm_.T_mid = get_field_in("T_mid").get_view<const Real **>();
  dry_atm_.p_mid = get_field_in("p_mid").get_view<const Real **>();
  dry_atm_.p_del = get_field_in("pseudo_density").get_view<const Real **>();


  // set wet/dry aerosol state data (interstitial aerosols only)
  for (int m = 0; m < mam_coupling::num_aero_modes(); ++m) {
    const char* int_nmr_field_name = mam_coupling::int_aero_nmr_field_name(m);
    wet_aero_.int_aero_nmr[m] = get_field_out(int_nmr_field_name).get_view<Real**>();
    //dry_aero_.int_aero_nmr[m] = buffer_.dry_int_aero_nmr[m];
    //MUST FIXME: We should compute dry mmr, not equate it to wet. This is WRONG!!
    dry_aero_.int_aero_nmr[m] = wet_aero_.int_aero_nmr[m];
    for (int a = 0; a < mam_coupling::num_aero_species(); ++a) {
      const char* int_mmr_field_name = mam_coupling::int_aero_mmr_field_name(m, a);
      if (strlen(int_mmr_field_name) > 0) {
        wet_aero_.int_aero_mmr[m][a] = get_field_out(int_mmr_field_name).get_view<Real**>();
        //dry_aero_.int_aero_mmr[m][a] = buffer_.dry_int_aero_mmr[m][a];
        //MUST FIXME: We should compute dry mmr, not equate it to wet. This is WRONG!!
        dry_aero_.int_aero_mmr[m][a] = wet_aero_.int_aero_mmr[m][a];
      }
    }
  }

  
}

// =========================================================================================
void MAMWetscav::run_impl(const double dt) {
  const auto policy =
      ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(ncol_, nlev_);

  /*Fortran code:
  call modal_aero_calcsize_sub(state%ncol, state%lchnk, state%q, state%pdel,
  dt, & !in qqcw, ptend, dgnumdry_m=dgncur_a) !inout

   ----subroutine modal_aero_calcsize_sub(ncol, lchnk, state_q, pdel, deltat,
  qqcw, ptend, do_adjust_in, & do_aitacc_transfer_in, list_idx_in,
  update_mmr_in, dgnumdry_m) */
  // mam4::CalcSizeProcess process_(aero_config_); //initiate MAM4xx calcsize
  // process

  /* ----------------------------------------------------------------------------------------
   * Compute particle size using the calcsize process
   * ----------------------------------------------------------------------------------------
   */

  // -- configure the process
  /*
   * -- NOTES: 1. Flags for the inter-mode particle transfer
   * (do_aitacc_transfer) and  size adjustment (do_adjust) are TRUE by default
   *                a. Size adjustment is only done by changing aerosol
   * numbers in the modes.
   *           2. Interstitial and cld borne aerosols (i.e. "tends") mmr will
   * be updated (update_mmr is TRUE by default)
   */

  int nspec_amode[ntot_amode_];
  int lspectype_amode[mam4::ndrop::maxd_aspectype][ntot_amode_];
  Real specdens_amode[mam4::ndrop::maxd_aspectype];
  int lmassptr_amode[mam4::ndrop::maxd_aspectype][ntot_amode_];
  Real spechygro[mam4::ndrop::maxd_aspectype];

  int numptr_amode[ntot_amode_];
  int mam_idx[ntot_amode_][mam4::ndrop::nspec_max];
  int mam_cnst_idx[ntot_amode_][mam4::ndrop::nspec_max];

  mam4::ndrop::get_e3sm_parameters(nspec_amode, lspectype_amode, lmassptr_amode,
                             numptr_amode, specdens_amode, spechygro, mam_idx,
                             mam_cnst_idx);

  Real inv_density[ntot_amode_][mam4::AeroConfig::num_aerosol_ids()] = {};
  Real num2vol_ratio_min[ntot_amode_]                                = {};
  Real num2vol_ratio_max[ntot_amode_]                                = {};
  Real num2vol_ratio_max_nmodes[ntot_amode_]                         = {};
  Real num2vol_ratio_min_nmodes[ntot_amode_]                         = {};
  Real num2vol_ratio_nom_nmodes[ntot_amode_]                         = {};
  Real dgnmin_nmodes[ntot_amode_]                                    = {};
  Real dgnmax_nmodes[ntot_amode_]                                    = {};
  Real dgnnom_nmodes[ntot_amode_]                                    = {};
  Real mean_std_dev_nmodes[ntot_amode_]                              = {};

  bool noxf_acc2ait[mam4::AeroConfig::num_aerosol_ids()]   = {};
  int n_common_species_ait_accum                           = {};
  int ait_spec_in_acc[mam4::AeroConfig::num_aerosol_ids()] = {};
  int acc_spec_in_ait[mam4::AeroConfig::num_aerosol_ids()] = {};
  mam4::modal_aero_calcsize::init_calcsize(
      inv_density, num2vol_ratio_min, num2vol_ratio_max,
      num2vol_ratio_max_nmodes, num2vol_ratio_min_nmodes,
      num2vol_ratio_nom_nmodes, dgnmin_nmodes, dgnmax_nmodes, dgnnom_nmodes,
      mean_std_dev_nmodes, noxf_acc2ait, n_common_species_ait_accum,
      ait_spec_in_acc, acc_spec_in_ait);

  // diagnostics for visible band summed over modes

  // Note: Need to compute inv density using indexing from e3sm
  for(int imode = 0; imode < ntot_amode_; ++imode) {
    const int nspec = nspec_amode[imode];
    for(int isp = 0; isp < nspec; ++isp) {
      const int idx           = lspectype_amode[isp][imode] - 1;
      inv_density[imode][isp] = 1.0 / specdens_amode[idx];
    }  // isp
  }    // imode

  /*view_2d state_q("state_q", nlev_, nvars_);

  for(int ilev=0; ilev<nlev_; ilev++){
    // 1d view of size ngas
    auto state_q_at_k = ekat::subview(state_q, ilev);
    mam_coupling::transfer_prognostics_to_work_arrays(progs,
                     k,
                     state_q_at_k.data(),
                     qqcw_q_at_k.data());
}*/
  



  // loop over atmosphere columns and compute aerosol particle size
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank();  // column index

      //state_q = mam_coupling::state_q_for_column(dry_aero_,icol);



        /*Real dgncur_c_kk[ntot_amode_]   = {};
        Real dgnumdry_m_kk[ntot_amode_] = {};
        //  Calculate aerosol size distribution parameters and aerosol water
        //  uptake
        // For prognostic aerosols
modal_aero_calcsize::modal_aero_calcsize_sub(
    state_q_kk, // in
    qqcw_k,     // in/out
    dt, do_adjust, do_aitacc_transfer, update_mmr, lmassptr_amode,
    numptr_amode,
    inv_density, // in
    num2vol_ratio_min, num2vol_ratio_max, num2vol_ratio_max_nmodes,
    num2vol_ratio_min_nmodes, num2vol_ratio_nom_nmodes, dgnmin_nmodes,
    dgnmax_nmodes, dgnnom_nmodes, mean_std_dev_nmodes,
    // outputs
    noxf_acc2ait, n_common_species_ait_accum, ait_spec_in_ */
      });
  

  /*
      ! Aerosol water uptake
      call t_startf('wateruptake')
      call modal_aero_wateruptake_dr(lchnk, ncol, state_q, temperature, pmid, &
     ! in cldn, dgncur_a, & ! in dgnumwet,  qaerwat, & ! inout wetdens=wetdens
     ) ! optional inout call t_stopf('wateruptake')

      ! skip wet deposition if nwetdep is non-positive
      if (nwetdep<1) return

      call calc_sfc_flux(rprdsh(:ncol,:),  state%pdel(:ncol,:),
     rprdshsum(:ncol))  ! output the last argument call
     calc_sfc_flux(rprddp(:ncol,:),  state%pdel(:ncol,:), rprddpsum(:ncol))  !
     output the last argument call calc_sfc_flux(evapcsh(:ncol,:),
     state%pdel(:ncol,:), evapcshsum(:ncol)) ! output the last argument call
     calc_sfc_flux(evapcdp(:ncol,:), state%pdel(:ncol,:), evapcdpsum(:ncol)) !
     output the last argument

      ! initiate variables
      qsrflx_mzaer2cnvpr(:,:,:) = 0.0_r8
      aerdepwetis(:,:)          = 0.0_r8
      aerdepwetcw(:,:)          = 0.0_r8
      qqcw_tmp(:,:)             = 0.0_r8
      ! below-cloud scavcoef = 0.0 for cloud-borne species
      scavcoefnv(:,:,0)         = 0.0_r8
      ! resuspension goes to a different phase or mode
      rtscavt_sv(:,:,:)         = 0.0_r8

      ! examine if there is precipitation falling from above in each grid
      call examine_prec_exist ( ncol,  state%pdel,      & ! in
           dep_inputs%prain,  dep_inputs%cmfdqr,& ! in
           dep_inputs%evapr,                    & ! in
           isprx                                ) ! out

      ! calculate the mass-weighted sol_factic for coarse mode species
      call set_f_act_coarse(      ncol,                           & ! in
           state%q,        ptend%q,        dt,             & ! in
           f_act_conv_coarse, f_act_conv_coarse_dust,      & ! out
           f_act_conv_coarse_nacl                          ) ! out

      mmode_loop_aa: do mtmp = 1, ntot_amode ! main loop over aerosol modes

         ! for mam4, do accum, aitken, pcarbon, then coarse
         ! so change the order of 3 and 4 here
         imode = mode_order_change(mtmp)

         ! loop over interstitial (1) and cloud-borne (2) forms
         !BSINGH (09/12/2014):Do cloudborne first for unified convection scheme
     so !that the resuspension of cloudborne can be saved then applied to
     interstitial (RCE) lphase_loop_aa:  do lphase = 2,1,-1  ! do cloudborne (2)
     first then interstitial (1)

            if (lphase == 1) then ! interstial aerosol
               call modal_aero_bcscavcoef_get( imode, ncol, isprx, dgnumwet, &
                    scavcoefnv(:,:,1), scavcoefnv(:,:,2) )
            endif
            call define_act_frac ( lphase,     imode,         & ! in
                 sol_facti, sol_factic, sol_factb, f_act_conv) ! out

            ! REASTER 08/12/2015 - changed ordering (mass then number) for
     prevap resuspend to coarse lspec_loop_aa: do lspec = 1,
     nspec_amode(imode)+2 ! loop over number + chem constituents + water

               call index_ordering (                        &
                    lspec, imode,  lphase,            & ! in
                    jaeronumb, jaeromass, jaerowater, & ! in
                    mm,    jnv, jnummaswtr            ) ! out

               if (mm <= 0 .or. jnummaswtr == jaerowater ) cycle  ! by pass wet
     aerosols

               ! mam_prevap_resusp_optcc values control the prevap_resusp
     calculations in wetdepa_v2: !     0 = no resuspension !   130 = non-linear
     resuspension of aerosol mass   based on scavenged aerosol mass !   230 =
     non-linear resuspension of aerosol number based on raindrop number !   the
     130 thru 230 all use the new prevap_resusp code block in subr wetdepa_v2
               !
               mam_prevap_resusp_optcc = mam_prevap_resusp_no

               if ( jnummaswtr == jaeromass ) then  ! dry mass
                  mam_prevap_resusp_optcc = mam_prevap_resusp_mass
               elseif ( jnummaswtr == jaeronumb .and. lphase == 1 .and. imode ==
     modeptr_coarse ) then ! number mam_prevap_resusp_optcc =
     mam_prevap_resusp_num endif

               ! set f_act_conv for interstitial (lphase=1) coarse mode species
               ! for the convective in-cloud, we conceptually treat the coarse
     dust and seasalt ! as being externally mixed, and apply ! f_act_conv =
     f_act_conv_coarse_dust/nacl to dust/seasalt ! number and sulfate are
     conceptually partitioned to the dust and seasalt ! on a mass basis, so the
     f_act_conv for number and sulfate are ! mass-weighted averages of the
     values used for dust/seasalt if ((lphase == 1) .and. (imode ==
     modeptr_coarse)) then f_act_conv = f_act_conv_coarse if (jnummaswtr ==
     jaeromass) then if (lmassptr_amode(lspec,imode) ==
     lptr_dust_a_amode(imode)) then f_act_conv = f_act_conv_coarse_dust elseif
     (lmassptr_amode(lspec,imode) == lptr_nacl_a_amode(imode)) then f_act_conv =
     f_act_conv_coarse_nacl endif endif endif

               lphase_jnmw_conditional: if (lphase == 1) then
                  ptend%lq(mm) = .true.
                  ! q_tmp reflects changes from modal_aero_calcsize and is the
     "most current" q q_tmp(1:ncol,:) = state%q(1:ncol,:,mm) +
     ptend%q(1:ncol,:,mm)*dt !Feed in the saved cloudborne mixing ratios from
     phase 2 qqcw_in(:,:) = qqcw_sav(:,:,lspec)

                  call wetdepa_v2( &
                       ncol, dt, state%pdel, & ! in dep_inputs%cmfdqr,
     dep_inputs%evapc, dlf, dep_inputs%conicw, & ! in dep_inputs%prain,
     dep_inputs%evapr, dep_inputs%totcond,    & ! in dep_inputs%cldt,
     dep_inputs%cldcu,                         & ! in dep_inputs%cldvcu,
     dep_inputs%cldvst,                      & ! in sol_factb, sol_facti,
     sol_factic,                          & ! in mam_prevap_resusp_optcc,
     .false., scavcoefnv(:,:,jnv), f_act_conv, & ! in q_tmp, qqcw_in(:,:), & !
     in fracis(:,:,mm), dqdt_tmp, iscavt,                          & ! out
                       icscavt, isscavt, bcscavt, bsscavt, rcscavt, rsscavt ) !
     out

                  ! resuspension goes to coarse mode
                  call calc_resusp_to_coarse(     ncol,   mm,     & ! in
                       mmtoo_prevap_resusp,    .true.,         & ! in
                       rcscavt,        rsscavt,                & ! in
                       dqdt_tmp,       rtscavt_sv              ) ! inout

                  ptend%q(1:ncol,:,mm) = ptend%q(1:ncol,:,mm) +
     dqdt_tmp(1:ncol,:)

                  call outfld( trim(cnst_name(mm))//'WET', dqdt_tmp(:,:), pcols,
     lchnk) call outfld( trim(cnst_name(mm))//'SIC', icscavt, pcols, lchnk) call
     outfld( trim(cnst_name(mm))//'SIS', isscavt, pcols, lchnk) call outfld(
     trim(cnst_name(mm))//'SBC', bcscavt, pcols, lchnk) call outfld(
     trim(cnst_name(mm))//'SBS', bsscavt, pcols, lchnk)

                  call calc_sfc_flux(dqdt_tmp(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx aerdepwetis(:ncol,mm) = sflx(:ncol)

                  call calc_sfc_flux(icscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx sflxic = sflx

                  call calc_sfc_flux(isscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name(mm))//'SFSIS', sflx,
     pcols, lchnk)

                  call calc_sfc_flux(bcscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name(mm))//'SFSBC', sflx,
     pcols, lchnk) sflxbc = sflx

                  call calc_sfc_flux(bsscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name(mm))//'SFSBS', sflx,
     pcols, lchnk)

                  ! here the prevap resuspension is in rcscavt & rsscavt and
     column integral is written to history !BSINGH(09/15/2014):Following two
     nested do-loops are new additions for unified convection
                  !BSINGH(09/15/2014):After these do-loops, code was added by
     RCE, the comments by RCE are kept as it is call
     calc_sfc_flux(rcscavt(:ncol,:), state%pdel(:ncol,:), sflx(:ncol)) ! output
     sflx sflxec = sflx

                  call calc_sfc_flux(rsscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name(mm))//'SFSES', sflx,
     pcols, lchnk)

                  ! apportion convective surface fluxes to deep and shallow conv
                  ! this could be done more accurately in subr wetdepa
                  ! since deep and shallow rarely occur simultaneously, and
     these !    fields are just diagnostics, this approximate method is adequate
                  ! only do this for interstitial aerosol, because conv clouds
     to not !    affect the stratiform-cloudborne aerosol call
     apportion_sfc_flux_deep ( ncol,              & ! in
                       rprddpsum,rprdshsum,evapcdpsum,evapcshsum,& ! in
                       sflxbc,             sflxec,               & ! in
                       sflxbcdp,           sflxecdp              ) ! out

                  call outfld( trim(cnst_name(mm))//'SFSBD', sflxbcdp, pcols,
     lchnk) ! when ma_convproc_intr is used, convective in-cloud wet removal is
     done there ! the convective (total and deep) precip-evap-resuspension
     includes in- and below-cloud ! contributions, so pass the below-cloud
     contribution to ma_convproc_intr qsrflx_mzaer2cnvpr(1:ncol,mm,1) = sflxec(
     1:ncol) qsrflx_mzaer2cnvpr(1:ncol,mm,2) = sflxecdp(1:ncol)

               elseif (lphase == 2) then lphase_jnmw_conditional
                  ! There is no cloud-borne aerosol water in the model, so this
     code block ! should NEVER execute for lspec = nspec_amode(m)+1 (i.e.,
     jnummaswtr = 2). ! The code only worked because the "do lspec" loop cycles
     when lspec = nspec_amode(m)+1, ! but that does not make the code correct.
                  fldcw => qqcw_get_field(pbuf,mm,lchnk)
                  qqcw_sav(1:ncol,:,lspec) = fldcw(1:ncol,:)  !RCE 2012/01/12

                  ! FIXME: Not sure if this is a bug or not as qqcw_tmp seem
     different ! from the previous call and qqcw_tmp is always zero. May need !
     further check.  - Shuaiqi Tang in refactoring for MAM4xx call wetdepa_v2( &
                       ncol, dt, state%pdel, & ! in dep_inputs%cmfdqr,
     dep_inputs%evapc, dlf, dep_inputs%conicw, & ! in dep_inputs%prain,
     dep_inputs%evapr, dep_inputs%totcond,      & ! in dep_inputs%cldt,
     dep_inputs%cldcu,                           & ! in dep_inputs%cldvcu,
     dep_inputs%cldvst,                        & ! in sol_factb, sol_facti,
     sol_factic,                            & ! in mam_prevap_resusp_optcc,
     .true., scavcoefnv(:,:,jnv), f_act_conv, & ! in fldcw, qqcw_tmp, & ! in
                       fracis_cw, dqdt_tmp, iscavt, & ! out icscavt, isscavt,
     bcscavt, bsscavt, rcscavt, rsscavt         ) ! out

                  ! resuspension goes to coarse mode
                  call calc_resusp_to_coarse(    ncol,   mm,      & ! in
                       mmtoo_prevap_resusp,   .false.,         & ! in
                       rcscavt,        rsscavt,                & ! in
                       dqdt_tmp,       rtscavt_sv              ) ! inout

                  fldcw(1:ncol,:) = fldcw(1:ncol,:) + dqdt_tmp(1:ncol,:) * dt

                  call calc_sfc_flux(dqdt_tmp(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFWET',
     sflx, pcols, lchnk) aerdepwetcw(:ncol,mm) = sflx(:ncol)

                  call calc_sfc_flux(icscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFSIC',
     sflx, pcols, lchnk)

                  call calc_sfc_flux(isscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFSIS',
     sflx, pcols, lchnk)

                  call calc_sfc_flux(bcscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFSBC',
     sflx, pcols, lchnk)

                  call calc_sfc_flux(bsscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFSBS',
     sflx, pcols, lchnk)

                  call calc_sfc_flux(rcscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFSEC',
     sflx, pcols, lchnk)

                  call calc_sfc_flux(rsscavt(:ncol,:), state%pdel(:ncol,:),
     sflx(:ncol)) ! output sflx call outfld( trim(cnst_name_cw(mm))//'SFSES',
     sflx, pcols, lchnk)

               endif lphase_jnmw_conditional

            enddo lspec_loop_aa  ! lspec = 1, nspec_amode(m)+2
         enddo lphase_loop_aa  ! lphase = 1, 2
      enddo mmode_loop_aa  ! m = 1, ntot_amode

      ! if the user has specified prescribed aerosol dep fluxes then
      ! do not set cam_out dep fluxes according to the prognostic aerosols
      if (.not.aerodep_flx_prescribed()) then
         call set_srf_wetdep(aerdepwetis, aerdepwetcw, cam_out)
      endif

      call wetdep_inputs_unset(dep_inputs)
  */
}

// =========================================================================================
}  // namespace scream
