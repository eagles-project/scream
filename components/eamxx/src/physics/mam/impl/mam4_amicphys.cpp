#include <mam4xx/aging.hpp>
#include <mam4xx/coagulation.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/nucleation.hpp>

namespace scream::impl {

#define MAX_FILENAME_LEN 256

using namespace mam4;

// number of constituents in gas chemistry "work arrays"
KOKKOS_INLINE_FUNCTION
constexpr int gas_pcnst() {
  constexpr int gas_pcnst_ = mam4::gas_chemistry::gas_pcnst;
  return gas_pcnst_;
}

// number of aerosol/gas species tendencies
KOKKOS_INLINE_FUNCTION
constexpr int nqtendbb() { return 4; }

// MAM4 aerosol microphysics configuration data
struct AmicPhysConfig {
  // these switches activate various aerosol microphysics processes
  bool do_cond;    // condensation (a.k.a gas-aerosol exchange)
  bool do_rename;  // mode "renaming"
  bool do_newnuc;  // gas -> aerosol nucleation
  bool do_coag;    // aerosol coagulation

  // configurations for specific aerosol microphysics
  mam4::GasAerExchProcess::ProcessConfig condensation;
  mam4::NucleationProcess::ProcessConfig nucleation;

  // controls treatment of h2so4 condensation in mam_gasaerexch_1subarea
  //    1 = sequential   calc. of gas-chem prod then condensation loss
  //    2 = simultaneous calc. of gas-chem prod and  condensation loss
  int gaexch_h2so4_uptake_optaa;

  // controls how nucleation interprets h2so4 concentrations
  int newnuc_h2so4_conc_optaa;
};

namespace {

// KOKKOS_INLINE_FUNCTION constexpr int nqtendaa() { return 5; } //old code
KOKKOS_INLINE_FUNCTION constexpr int nqtendaa() { return 4; }
KOKKOS_INLINE_FUNCTION constexpr int nqqcwtendaa() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int nqqcwtendbb() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_cond() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_rnam() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_nnuc() { return 2; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_coag() { return 3; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_cond_only() { return 4; }
KOKKOS_INLINE_FUNCTION constexpr int iqqcwtend_rnam() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int n_agepair() { return 1; }
// FIXME: Add comments why maxareas is 3!!!
KOKKOS_INLINE_FUNCTION constexpr int maxsubarea() { return 3; }
// In state_q, we have 40 species, the gasses and aerosols starts after the
//  9th index, so loffset is 9
KOKKOS_INLINE_FUNCTION constexpr int loffset() { return 9; }
// number of gases used in aerosol microphysics
KOKKOS_INLINE_FUNCTION constexpr int max_gas() { return 2; }

// leave number mix-ratios unchanged (#/kmol-air)
KOKKOS_INLINE_FUNCTION Real fcvt_num() { return 1; }
// factor for converting aerosol water mix-ratios from (kg/kg) to (mol/mol)
KOKKOS_INLINE_FUNCTION Real fcvt_wtr() {
  return haero::Constants::molec_weight_dry_air /
         haero::Constants::molec_weight_h2o;
}

KOKKOS_INLINE_FUNCTION constexpr int lmapcc_val_nul() { return 0; }
KOKKOS_INLINE_FUNCTION constexpr int lmapcc_val_gas() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int lmapcc_val_aer() { return 2; }
KOKKOS_INLINE_FUNCTION constexpr int lmapcc_val_num() { return 3; }
KOKKOS_INLINE_FUNCTION int lmapcc_all(const int index) {
  static constexpr int lmapcc_all_[gas_pcnst()] = {
      lmapcc_val_nul(), lmapcc_val_nul(), lmapcc_val_gas(), lmapcc_val_nul(),
      lmapcc_val_nul(), lmapcc_val_gas(), lmapcc_val_aer(), lmapcc_val_aer(),
      lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_aer(),
      lmapcc_val_aer(), lmapcc_val_num(), lmapcc_val_aer(), lmapcc_val_aer(),
      lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_num(), lmapcc_val_aer(),
      lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_aer(),
      lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_num(), lmapcc_val_aer(),
      lmapcc_val_aer(), lmapcc_val_aer(), lmapcc_val_num()};
  return lmapcc_all_[index];
}

// Indices of aerosol number for the arrays dimensioned gas_pcnst
KOKKOS_INLINE_FUNCTION int numptr_amode_gas_pcnst(const int mode) {
  static constexpr int numptr_amode_gas_pcnst_[AeroConfig::num_modes()] = {
      13, 18, 26, 30};
  return numptr_amode_gas_pcnst_[mode];
}

// Where lmapcc_val_gas are defined in lmapcc_all
KOKKOS_INLINE_FUNCTION int lmap_gas(const int mode) {
  static constexpr int lmap_gas_[AeroConfig::num_modes()] = {5, 2};
  return lmap_gas_[mode];
}

// Returns index of aerosol numbers in gas_pcnst array
KOKKOS_INLINE_FUNCTION int lmap_num(const int mode) {
  return numptr_amode_gas_pcnst(mode);
}
// aerosol mapping for aerosol microphysics
// NOTE: it is different from "lmassptr_amode_gas_pcnst" as
//       amicphys adds aerosol species in a special order that is different from
//       lmassptr_amode_gas_pcnst
KOKKOS_INLINE_FUNCTION int lmap_aer(const int iaer, const int mode) {
  static constexpr int
      lmap_aer_[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] = {
          {8, 15, 24, -1},  {6, 14, 21, -1},  {7, -1, 23, 27},  {9, -1, 22, 28},
          {11, 16, 20, -1}, {10, -1, 19, -1}, {12, 17, 25, 29},
      };
  return lmap_aer_[iaer][mode];
}

// conversion factor for aerosols
// NOTE: The following array has a special order to match amicphys
KOKKOS_INLINE_FUNCTION Real fcvt_aer(const int iaer) {
  static constexpr Real fcvt_aer_[AeroConfig::num_aerosol_ids()] = {
      8.000000000000000E-002, 1, 8.000000000000000E-002, 1, 1, 1, 1};
  return fcvt_aer_[iaer];
}

// Number of differently tagged secondary-organic aerosol species
KOKKOS_INLINE_FUNCTION constexpr int nsoa() { return 1; }

// conversion factor for gases
KOKKOS_INLINE_FUNCTION Real fcvt_gas(const int gas_id) {
  // mw to use for soa
  constexpr Real mwuse_soa = 150;
  // molecular weight of the gas
  Real mw_gas = mam4::gas_chemistry::adv_mass[lmap_gas(gas_id)];
  // denominator
  Real denom = mw_gas;
  // special case for soa
  if(gas_id < nsoa()) denom = mwuse_soa;
  return mw_gas / denom;
}

// Indices of aerosol mass for the arrays dimensioned gas_pcnst
KOKKOS_INLINE_FUNCTION int lmassptr_amode_gas_pcnst(const int aero_id,
                                                    const int mode) {
  static constexpr int lmassptr_amode_gas_pcnst_[AeroConfig::num_aerosol_ids()]
                                                [AeroConfig::num_modes()] = {
                                                    {6, 14, 19, 27},
                                                    {7, 15, 20, 28},
                                                    {8, 16, 21, 29},
                                                    {9, 17, 22, -6},
                                                    {10, -6, 23, -6},
                                                    {11, -6, 24, -6},
                                                    {12, -6, 25, -6}};
  return lmassptr_amode_gas_pcnst_[aero_id][mode];
}

KOKKOS_INLINE_FUNCTION
void subarea_partition_factors(
    const Real
        q_int_cell_avg,  // in grid cell mean interstitial aerosol mixing ratio
    const Real
        q_cbn_cell_avg,  // in grid cell mean cloud-borne  aerosol mixing ratio
    const Real fcldy,    // in  cloudy fraction of the grid cell
    const Real fclea,    // in clear  fraction of the grid cell
    Real &part_fac_q_int_clea,  // out
    Real &part_fac_q_int_cldy)  // out
{
  // Calculate mixing ratios of each subarea

  // cloud-borne,  cloudy subarea
  const Real tmp_q_cbn_cldy = q_cbn_cell_avg / fcldy;
  // interstitial, cloudy subarea
  const Real tmp_q_int_cldy =
      haero::max(0.0, ((q_int_cell_avg + q_cbn_cell_avg) - tmp_q_cbn_cldy));
  // interstitial, clear  subarea
  const Real tmp_q_int_clea = (q_int_cell_avg - fcldy * tmp_q_int_cldy) / fclea;

  // Calculate the corresponding paritioning factors for interstitial aerosols
  // using the above-derived subarea mixing ratios plus the constraint that
  // the cloud fraction weighted average of subarea mean need to match grid box
  // mean.

  // *** question ***
  //    use same part_fac_q_int_clea/cldy for everything ?
  //    use one for number and one for all masses (based on total mass) ?
  //    use separate ones for everything ?
  // maybe one for number and one for all masses is best,
  //    because number and mass have different activation fractions
  // *** question ***

  Real tmp_aa = haero::max(1.e-35, tmp_q_int_clea * fclea) /
                haero::max(1.e-35, q_int_cell_avg);
  tmp_aa = haero::max(0.0, haero::min(1.0, tmp_aa));

  part_fac_q_int_clea = tmp_aa / fclea;
  part_fac_q_int_cldy = (1.0 - tmp_aa) / fcldy;
}

KOKKOS_INLINE_FUNCTION
void setup_subareas(int kk, const Real cld,             // in //FIXME: remove kk
                    int &nsubarea, int &ncldy_subarea,  // out
                    int &jclea, int &jcldy,             // out
                    bool (&iscldy_subarea)[(maxsubarea())],  // out
                    Real (&afracsub)[maxsubarea()],          // out
                    Real &fclea, Real &fcldy)                // out
{
  //--------------------------------------------------------------------------------------
  // Purpose: Determine the number of sub-areas and their fractional areas.
  //          Assign values to some bookkeeping variables.
  //--------------------------------------------------------------------------------------

  // cld: cloud fraction in the grid cell [unitless]
  // nsubarea: total # of subareas to do calculations for
  // ncldy_subarea: total # of cloudy subareas
  // jclea, jcldy: indices of the clear and cloudy subareas
  // iscldy_subarea(maxsubarea): whether a subarea is cloudy
  // afracsub(maxsubarea): area fraction of each active subarea[unitless]
  // fclea, fcldy: area fraction of clear/cloudy subarea [unitless]

  // BAD CONSTANT
  //  Cloud chemistry is only active when cld(i,kk) >= 1.0e-5
  //  It may be that the macrophysics has a higher threshold than this
  constexpr Real fcld_locutoff = 1.0e-5;

  // BAD CONSTANT
  //  Grid cells with cloud fraction larger than this cutoff is considered to be
  //  overcast
  constexpr Real fcld_hicutoff = 0.999;

  // if cloud fraction ~= 0, the grid-cell has a single clear  sub-area
  // (nsubarea = 1) if cloud fraction ~= 1, the grid-cell has a single cloudy
  // sub-area (nsubarea = 1) otherwise, the grid-cell has a clear and a cloudy
  // sub-area (nsubarea = 2)

  if(cld < fcld_locutoff) {
    fcldy         = 0;
    nsubarea      = 1;
    ncldy_subarea = 0;
    jclea         = 1;
    jcldy         = 0;
  } else if(cld > fcld_hicutoff) {
    fcldy         = 1;
    nsubarea      = 1;
    ncldy_subarea = 1;
    jclea         = 0;
    jcldy         = 1;
  } else {
    fcldy         = cld;
    nsubarea      = 2;
    ncldy_subarea = 1;
    jclea         = 1;
    jcldy         = 2;
  }

  fclea = 1.0 - fcldy;

  // Set up a logical array to indicate whether the subareas are clear or cloudy
  // and init area fraction array
  for(int jsub = 0; jsub < maxsubarea(); ++jsub) {
    iscldy_subarea[jsub] = false;
    afracsub[jsub]       = 0;
  }

  // jcldy>0 can be 1 or 2, so iscldy_subarea(1) or iscldy_subarea(2) is true
  if(jcldy > 0) iscldy_subarea[jcldy] = true;
  // Save the area fractions to an array
  // jclea can only be 1 if jclea > 0, so afracsub (1) is set to fclea
  if(jclea > 0) afracsub[jclea] = fclea;
  // jcldy can be 1 or 2, so afracsub(1) or afracsub(2) is set to fcldy
  if(jcldy > 0) afracsub[jcldy] = fcldy;

}  // setup_subareas

KOKKOS_INLINE_FUNCTION
void set_subarea_rh(const int &ncldy_subarea, const int &jclea,  // in
                    const int &jcldy,                            // in
                    const Real (&afracsub)[maxsubarea()],        // in
                    const Real &relhumgcm,                       // in
                    Real (&relhumsub)[maxsubarea()])             // out
{
  //----------------------------------------------------------------------------
  // Purpose: Set relative humidity in subareas.
  //----------------------------------------------------------------------------

  // ncldy_subarea         :# of cloudy subareas
  // jclea, jcldy          :indices of clear and cloudy subareas
  // afracsub(maxsubarea)  :area fraction in subareas [unitless]
  // relhumgcm             :grid cell mean relative humidity [unitless]
  // relhumsub(maxsubarea): relative humidity in subareas [unitless]

  if(ncldy_subarea <= 0) {
    // Entire grid cell is cloud-free. RH in subarea = grid cell mean.
    // This is clear cell, rehumsub(0),rehumsub(1) and rehumsub(3) are relhumgcm
    for(int jsub = 0; jsub < maxsubarea(); ++jsub) relhumsub[jsub] = relhumgcm;
  } else {
    // Grid cell has a cloudy subarea. Set RH in that part to 1.0.
    // jcldy can be 1 or 2 here.
    // If jcldy is 1: relhumsub[1] is 1.0 (fully cloudy cell)
    // if jcldy is 2: relhumsub[2] is 1.0. In this case jclea is >0,
    //               so relhumsub[1] is set in if condition below
    relhumsub[jcldy] = 1;

    // If the grid cell also has a clear portion, back out the subarea RH from
    // the grid-cell mean RH and cloud fraction.
    if(jclea > 0) {
      // jclea is > 0 only for partly cloudy cell. In this case
      // jclea is 1, so relhumsub[1] is set here.
      Real relhum_tmp  = (relhumgcm - afracsub[jcldy]) / afracsub[jclea];
      relhumsub[jclea] = mam4::utils::min_max_bound(0, 1, relhum_tmp);
    }
  }
}  // set_subarea_rh

KOKKOS_INLINE_FUNCTION
void compute_qsub_from_gcm_and_qsub_of_other_subarea(
    int kk,  // FIXME remove kk
    const bool (&lcompute)[gas_pcnst()], const Real &f_a,
    const Real &f_b,                  // in
    const Real (&qgcm)[gas_pcnst()],  // in
    const int &jclea, const int &jcldy,
    Real (&qsub_a)[gas_pcnst()][maxsubarea()],  // inout
    Real (&qsub_b)[gas_pcnst()][maxsubarea()])  // inout
{
  //-----------------------------------------------------------------------------------------
  // Purpose: Calculate the value of qsub_b assuming qgcm is a weighted average
  // defined as
  //          qgcm = f_a*qsub_a + f_b*qsub_b.
  //-----------------------------------------------------------------------------------------

  // f_a, f_b      // area fractions [unitless] of subareas
  // qgcm(ncnst)   // grid cell mean (known)
  // qsub_a(ncnst) // value in subarea A (known, but might get adjusted)
  // qsub_b(ncnst) // value in subarea B (to be calculated here)

  // Here we populate qsub for subarea index 2 (i.e. jcldy is 2 here)
  //  and adjust subarea index 1(i.e., jclea is 1 here) if needed.
  for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
    if(kk == 48) printf("HERE1\n");
    if(lcompute[icnst]) {
      if(kk == 48) printf("HERE2\n");
      // Calculate qsub_b
      EKAT_KERNEL_ASSERT_MSG(
          f_b != 0,
          "Error! compute_qsub_from_gcm_and_qsub_of_other_subarea - f_b is "
          "zero\n");
      qsub_b[icnst][jcldy] = (qgcm[icnst] - f_a * qsub_a[icnst][jclea]) / f_b;
      if(kk == 48)
        printf("HERE3:%0.15e,%0.15e,%0.15e,%0.15e,%0.15e,%i,%i,%i\n",
               qsub_b[icnst][jcldy], qgcm[icnst], f_a, f_b,
               qsub_a[icnst][jclea], icnst, jclea, jcldy);

      // Check that this does not produce a negative value.
      // If so, set qsub_b to zero and adjust the value of qsub_a.
      if(qsub_b[icnst][jcldy] < 0) {
        qsub_b[icnst][jcldy] = 0;
        EKAT_KERNEL_ASSERT_MSG(
            f_a != 0,
            "Error! compute_qsub_from_gcm_and_qsub_of_other_subarea - f_a is "
            "zero\n");
        qsub_a[icnst][jclea] = qgcm[icnst] / f_a;
      }
    }
  }
}  // compute_qsub_from_gcm_and_qsub_of_other_subarea

KOKKOS_INLINE_FUNCTION
void set_subarea_qnumb_for_cldbrn_aerosols(
    const int &jclea, const int &jcldy, const Real &fcldy,  // in
    const Real (&qqcwgcm)[gas_pcnst()],                     // in
    Real (&qqcwsub)[gas_pcnst()][maxsubarea()])             // inout
{
  //-----------------------------------------------------------------------------------------
  // Purpose: Set the number mixing ratios of cloud-borne aerosols in subareas:
  //          - zero in clear air;
  //          - grid-cell-mean divided by cloud-fraction in the cloudy subarea.
  //          This is done for all lognormal modes.
  //-----------------------------------------------------------------------------------------

  // jclea, jcldy              : indices of subareas
  // fcldy                     : area fraction [unitless] of the cloudy subarea
  // qqcwgcm(ncnst)            : grid cell mean (unit does not matter for this
  //                             subr.)
  // qqcwsub(ncnst,maxsubarea) : values in subareas (unit does not matter
  //                             for this subr.)

  //----------------------------------------------------------------
  // Here jclea ==1 and jcldy==2
  for(int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    const int icnst       = numptr_amode_gas_pcnst(imode);
    qqcwsub[icnst][jclea] = 0;
    EKAT_KERNEL_ASSERT_MSG(
        fcldy != 0,
        "Error! set_subarea_qnumb_for_cldbrn_aerosols - fcldy is "
        "zero\n");
    qqcwsub[icnst][jcldy] = qqcwgcm[icnst] / fcldy;
    //----------------------------------------------------------------
  }

}  // set_subarea_qnumb_for_cldbrn_aerosols

KOKKOS_INLINE_FUNCTION
void set_subarea_qmass_for_cldbrn_aerosols(
    const int &jclea, const int &jcldy,          // in
    const Real &fcldy,                           // in
    const Real (&qqcwgcm)[gas_pcnst()],          // in
    Real (&qqcwsub)[gas_pcnst()][maxsubarea()])  // inout
{
  //-----------------------------------------------------------------------------------------
  // Purpose: Set the mass mixing ratios of cloud-borne aerosols in subareas:
  //          - zero in clear air;
  //          - grid-cell-mean/cloud-fraction in the cloudy subarea.
  //          This is done for all lognormal modes and all chemical species.
  //-----------------------------------------------------------------------------------------
  // jclea, jcldy              : subarea indices fcldy : area
  //                             fraction [unitless] of the cloudy subarea
  // qqcwgcm(ncnst)            : grid cell mean (unit does not matter for this
  //                             subr.)
  // qqcwsub(ncnst,maxsubarea) : values in subareas (unit does not matter for
  //                             this subr.)

  //----------------------------------------------------------------
  // Here jclea ==1 and jcldy==2

  // loop thru all modes
  for(int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    // loop thru all species in a mode
    for(int ispec = 0; ispec < mam4::num_species_mode(imode); ++ispec) {
      const int icnst = lmassptr_amode_gas_pcnst(ispec, imode);

      qqcwsub[icnst][jclea] = 0;
      EKAT_KERNEL_ASSERT_MSG(
          fcldy != 0,
          "Error! set_subarea_qmass_for_cldbrn_aerosols - fcldy is "
          "zero\n");
      qqcwsub[icnst][jcldy] = qqcwgcm[icnst] / fcldy;
    }  // ispec - species loop
  }    // imode - mode loop
       //----------------------------------------------------------------
}  // set_subarea_qmass_for_cldbrn_aerosols

KOKKOS_INLINE_FUNCTION
void get_partition_factors(const Real &qgcm_intrst,               // in
                           const Real &qgcm_cldbrn,               // in
                           const Real &fcldy, const Real &fclea,  // in
                           Real &factor_clea, Real &factor_cldy)  // out
{
  //------------------------------------------------------------------------------------
  // Purpose: Calculate the partitioning factors for distributing interstitial
  // aerosol
  //          mixing ratios to cloudy and clear subareas in a grid box.
  //          The partitioning factors depend on the grid cell mean mixing
  //          ratios of both interstitial and cloud-borne aerosols.
  //------------------------------------------------------------------------------------

  // qgcm_intrst  : grid cell mean interstitial aerosol mixing ratio
  // qgcm_cldbrn  : grid cell mean cloud-borne aerosol mixing ratio
  //
  // fcldy        : cloudy fraction of the grid cell [unitless]
  // fclea        : clear  fraction of the grid cell [unitless]
  //
  // factor_clea  : partitioning factor for clear  subarea
  // factor_cldy  : partitioning factor for cloudy subarea

  // Calculate subarea-mean mixing ratios

  EKAT_KERNEL_ASSERT_MSG(fcldy != 0,
                         "Error! get_partition_factors - fcldy is "
                         "zero\n");
  // cloud-borne,  cloudy subarea
  const Real tmp_q_cldbrn_cldy = qgcm_cldbrn / fcldy;

  // interstitial, cloudy subarea
  const Real tmp_q_intrst_cldy =
      haero::max(0, ((qgcm_intrst + qgcm_cldbrn) - tmp_q_cldbrn_cldy));

  EKAT_KERNEL_ASSERT_MSG(fclea != 0,
                         "Error! get_partition_factors - fclea is "
                         "zero\n");
  // interstitial, clear  subarea
  const Real tmp_q_intrst_clea =
      (qgcm_intrst - fcldy * tmp_q_intrst_cldy) / fclea;

  // Calculate the corresponding paritioning factors for interstitial
  // aerosols using the above-derived subarea-mean mixing ratios plus the
  // constraint that the cloud fraction weighted average of subarea mean
  // need to match grid box mean. Note that this subroutine is designed for
  // partially cloudy grid cells, hence both fclea and fcldy are assumed to
  // be nonzero.

  constexpr Real eps = 1.e-35;  // BAD CONSTANT
  Real clea2gcm_ratio =
      haero::max(eps, tmp_q_intrst_clea * fclea) / haero::max(eps, qgcm_intrst);
  clea2gcm_ratio = haero::max(0, haero::min(1, clea2gcm_ratio));

  factor_clea = clea2gcm_ratio / fclea;
  factor_cldy = (1 - clea2gcm_ratio) / fcldy;
}  // get_partition_factors

KOKKOS_INLINE_FUNCTION
void set_subarea_qnumb_for_intrst_aerosols(
    const int &jclea, const int &jcldy, const Real &fclea,  // in
    const Real &fcldy, const Real (&qgcm)[gas_pcnst()],     // in
    const Real (&qqcwgcm)[gas_pcnst()],                     // in
    const Real (&qgcmx)[gas_pcnst()],                       // in
    Real (&qsubx)[gas_pcnst()][maxsubarea()])               // inout
{
  //-----------------------------------------------------------------------------------------
  // Purpose: Set the number mixing ratios of interstitial aerosols in subareas.
  //          Interstitial aerosols can exist in both cloudy and clear subareas,
  //          so a grid cell mean needs to be partitioned. Different lognormal
  //          modes are partitioned differently based on the mode-specific
  //          number mixing ratios.
  //-----------------------------------------------------------------------------------------

  // jclea, jcldy  : subarea indices
  // fclea, fcldy  : area fraction [unitless] of the clear and cloudy subareas
  // qgcm   (ncnst): grid cell mean, interstitial constituents (unit does not
  //                 matter)
  // qqcwgcm(ncnst): grid cell mean, cloud-borne  constituents (unit
  //                 does not matter)

  // qgcmx  (ncnst): grid cell mean, interstitial constituents (unit does not
  //                 matter)
  // qsubx(ncnst,maxsubarea): subarea mixing ratios of interst. constituents
  //                          (unit does not matter as long as they are
  //                          consistent with the grid cell mean values)

  // Note: qgcm and qqcwgcm are used for calculating the patitioning factors.
  // qgcmx is the actual grid cell mean that is partitioned into qsubx.

  for(int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    // calculate partitioning factors

    // grid cell mean of interstitial aerosol mixing ratio of a single mode
    const Real qgcm_intrst = qgcm[numptr_amode_gas_pcnst(imode)];

    // grid cell mean of cloud-borne  aerosol mixing ratio of a single mode
    const Real qgcm_cldbrn = qqcwgcm[numptr_amode_gas_pcnst(imode)];

    Real factor_clea;  // partitioning factor for clear  subarea [unitless]
    Real factor_cldy;  // partitioning factor for cloudy subarea [unitless]
    get_partition_factors(qgcm_intrst, qgcm_cldbrn, fcldy, fclea,  // in
                          factor_clea, factor_cldy);               // out

    // apply partitioning factors
    const int icnst = numptr_amode_gas_pcnst(imode);

    qsubx[icnst][jclea] = qgcmx[icnst] * factor_clea;
    qsubx[icnst][jcldy] = qgcmx[icnst] * factor_cldy;

  }  // imode

}  // set_subarea_qnumb_for_intrst_aerosols

KOKKOS_INLINE_FUNCTION
void set_subarea_qmass_for_intrst_aerosols(
    const int &jclea, const int &jcldy, const Real &fclea,  // in
    const Real &fcldy, const Real (&qgcm)[gas_pcnst()],     // in
    const Real (&qqcwgcm)[gas_pcnst()],                     // in
    const Real (&qgcmx)[gas_pcnst()],                       // in
    Real (&qsubx)[gas_pcnst()][maxsubarea()])               // inout
{
  //-----------------------------------------------------------------------------------------
  // Purpose: Set the mass mixing ratios of interstitial aerosols in subareas.
  //          Interstitial aerosols can exist in both cloudy and clear subareas,
  //          so a grid cell mean needs to be partitioned. Different lognormal
  //          modes are partitioned differently based on the mode-specific
  //          mixing ratios. All species in the same mode are partitioned the
  //          same way, consistent with the internal mixing assumption used in
  //          MAM.
  //-----------------------------------------------------------------------------------------

  // jclea, jcldy   : subarea indices
  // fclea, fcldy   : area fraction [unitless] of the clear and cloudy subareas
  // qgcm   (ncnst) : grid cell mean, interstitial constituents (unit does not
  //                  matter)
  // qqcwgcm(ncnst) : grid cell mean, cloud-borne  constituents (unit
  //                  does not matter)

  // qgcmx  (ncnst) : grid cell mean, interstitial constituents (unit does not
  //                  matter)
  // qsubx(ncnst,maxsubarea): subarea mixing ratios of interst.
  //                          constituents(unit does not matter as long as they
  //                          are consistent with the grid cell mean values)

  // Note: qgcm and qqcwgcm are used for calculating the patitioning factors.
  // qgcmx is the actual grid cell mean that is partitioned into qsubx.

  for(int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    // calculcate partitioning factors

    // grid cell mean of interstitial aerosol mixing ratio of a single mode
    Real qgcm_intrst = 0;

    // grid cell mean of cloud-borne  aerosol mixing ratio of a single mode
    Real qgcm_cldbrn = 0;

    // loop thru all species in a mode
    for(int ispec = 0; ispec < mam4::num_species_mode(imode); ++ispec) {
      qgcm_intrst = qgcm_intrst + qgcm[lmassptr_amode_gas_pcnst(ispec, imode)];
      qgcm_cldbrn =
          qgcm_cldbrn + qqcwgcm[lmassptr_amode_gas_pcnst(ispec, imode)];
    }

    Real factor_clea;  // partitioning factor for clear  subarea [unitless]
    Real factor_cldy;  // partitioning factor for cloudy subarea [unitless]
    get_partition_factors(qgcm_intrst, qgcm_cldbrn, fcldy, fclea,  // in
                          factor_clea, factor_cldy);               // out

    // apply partitioning factors
    // Here jclea==1 and jcldy==2
    for(int ispec = 0; ispec < mam4::num_species_mode(imode); ++ispec) {
      const int icnst     = lmassptr_amode_gas_pcnst(ispec, imode);
      qsubx[icnst][jclea] = qgcmx[icnst] * factor_clea;
      qsubx[icnst][jcldy] = qgcmx[icnst] * factor_cldy;
    }  // ispec
  }    // imode

}  // set_subarea_qmass_for_intrst_aerosols

KOKKOS_INLINE_FUNCTION
void set_subarea_gases_and_aerosols(
    int kk, const int &nsubarea, const int &jclea,                       // in
    const int &jcldy,                                                    // in
    const Real &fclea, const Real &fcldy,                                // in
    const Real (&qgcm1)[gas_pcnst()], const Real (&qgcm2)[gas_pcnst()],  // in
    const Real (&qqcwgcm2)[gas_pcnst()],                                 // in
    const Real (&qgcm3)[gas_pcnst()],                                    // in
    const Real (&qqcwgcm3)[gas_pcnst()],                                 // in
    Real (&qsub1)[gas_pcnst()][maxsubarea()],                            // out
    Real (&qsub2)[gas_pcnst()][maxsubarea()],                            // out
    Real (&qqcwsub2)[gas_pcnst()][maxsubarea()],                         // out
    Real (&qsub3)[gas_pcnst()][maxsubarea()],                            // out
    Real (&qqcwsub3)[gas_pcnst()][maxsubarea()])                         // out
{
  //------------------------------------------------------------------------------------------------
  // Purpose: Partition grid cell mean mixing ratios to clear/cloudy subareas.
  //------------------------------------------------------------------------------------------------
  // nsubarea: # of active subareas in the current grid cell
  // jclea, jcldy: indices of the clear and cloudy subareas
  // fclea, fcldy: area fractions of the clear and cloudy subareas [unitless]

  // The next set of argument variables are tracer mixing ratios.
  //  - The units are different for gases, aerosol number, and aerosol mass.
  //    The exact units do not matter for this subroutine, as long as the
  //    grid cell mean values ("gcm") and the corresponding subarea values
  //    ("sub") have the same units.
  //  - q* and qqcw* are correspond to the interstitial and cloud-borne
  //    species, respectively
  //  - The numbers 1-3 correspond to different locations in the host model's
  //    time integration loop.

  // Grid cell mean mixing ratios
  // qgcm1(ncnst), qgcm2(ncnst), qqcwgcm2(ncnst), qgcm3(ncnst),
  // qqcwgcm3(ncnst)

  // Subarea mixing ratios
  // qsub1(ncnst,maxsubarea), qsub2(ncnst,maxsubarea),
  // qsub3(ncnst,maxsubarea), qqcwsub2(ncnst,maxsubarea)
  // qqcwsub3(ncnst,maxsubarea)
  //----

  //------------------------------------------------------------------------------------
  // Initialize mixing ratios in subareas before the aerosol microphysics
  // calculations
  //------------------------------------------------------------------------------------
  // FIXME:Should we set jsub==0 a special value (like NaNs) so that it is never
  // used??
  for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
    for(int jsub = 0; jsub < maxsubarea(); ++jsub) {
      // Gases and interstitial aerosols
      qsub1[icnst][jsub] = 0;
      qsub2[icnst][jsub] = 0;
      qsub3[icnst][jsub] = 0;

      // Cloud-borne aerosols
      qqcwsub2[icnst][jsub] = 0;
      qqcwsub3[icnst][jsub] = 0;
    }
  }
  //---------------------------------------------------------------------------------------------------
  // Determine which category the current grid cell belongs to: partly cloudy,
  // all cloudy, or all clear
  //---------------------------------------------------------------------------------------------------
  const bool grid_cell_has_only_clea_area =
      ((jclea == 1) && (jcldy == 0) && (nsubarea == 1));
  const bool grid_cell_has_only_cldy_area =
      ((jclea == 0) && (jcldy == 1) && (nsubarea == 1));
  const bool gird_cell_is_partly_cldy =
      (jclea > 0) && (jcldy > 0) && (jclea + jcldy == 3) && (nsubarea == 2);

  // Sanity check
  if((!grid_cell_has_only_clea_area) && (!grid_cell_has_only_cldy_area) &&
     (!gird_cell_is_partly_cldy)) {
    EKAT_KERNEL_ASSERT_MSG(true,
                           "Error! modal_aero_amicphys - bad jclea, jcldy, "
                           "nsubarea, jclea, jcldy, nsubarea\n");
  }

  //*************************************************************************************************
  // Category I: grid cell is either all clear or all cloudy. Copy the grid
  // cell mean values.
  //*************************************************************************************************
  if(grid_cell_has_only_clea_area || grid_cell_has_only_cldy_area) {
    // For fully clear and cloudy cells, we populate only 1st index of subarea
    // for all output vars
    //  Makes sense as there is only 1 subarea for these cases.
    // FIXME: Should we fill in NaNs for the 0th and 2nd index??
    constexpr int jsub = 1;
    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      // copy all gases and aerosols
      if(lmapcc_all(icnst) > 0) {
        qsub1[icnst][jsub] = qgcm1[icnst];
        qsub2[icnst][jsub] = qgcm2[icnst];
        qsub3[icnst][jsub] = qgcm3[icnst];

        qqcwsub2[icnst][jsub] = qqcwgcm2[icnst];
        qqcwsub3[icnst][jsub] = qqcwgcm3[icnst];
      }
    }
    if(kk == 48)
      printf("qsub1_1:%0.15e, %i, %i, %i\n", qsub1[2][1], jclea, jcldy,
             nsubarea);
  }
  //*************************************************************************************************
  // Category II: partly cloudy grid cell. Tracer mixing ratios are generally
  // assumed different in clear and cloudy subareas.  This is primarily
  // because the interstitial aerosol mixing ratios are assumed to be lower
  // in the cloudy sub-area than in the clear sub-area, as much of the
  // aerosol is activated in the cloudy sub-area.
  //*************************************************************************************************
  else if(gird_cell_is_partly_cldy) {
    //===================================
    // Set gas mixing ratios in subareas
    //===================================
    //------------------------------------------------------------------------------------------
    // Before gas chemistry, gas mixing ratios are assumed to be the same in
    // all subareas, i.e., they all equal the grid cell mean.
    //------------------------------------------------------------------------------------------

    // NOTE: In this "else if" case jclea == 1 and jcldy == 2

    bool cnst_is_gas[gas_pcnst()] = {};
    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      cnst_is_gas[icnst] = (lmapcc_all(icnst) == lmapcc_val_gas());
    }

    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      if(cnst_is_gas[icnst]) {
        // For gases, assume both 1 and 2 subareas have grid mean values
        for(int jsub = 1; jsub <= nsubarea; ++jsub) {
          qsub1[icnst][jsub] = qgcm1[icnst];
        }
      }
    }
    // qsub1 is fully populated for gasses
    if(kk == 48)
      printf("qsub1_2:%0.15e,%0.15e, %i\n", qsub1[2][2], qsub1[2][2], nsubarea);
    //------------------------------------------------------------------------------------------
    // After gas chemistry, still assume gas mixing ratios are the same in all
    // subareas.
    //------------------------------------------------------------------------------------------

    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      if(cnst_is_gas[icnst]) {
        // For gases, assume both 1 and 2 subareas have grid mean values
        for(int jsub = 1; jsub <= nsubarea; ++jsub) {
          qsub2[icnst][jsub] = qgcm2[icnst];
        }
      }
    }
    // if(kk == 48) printf("qsub1_3:%0.15e\n", qsub2[2][2]);
    //  qsub2 is fully populated for gasses
    //----------------------------------------------------------------------------------------
    //   After cloud chemistry, gas and aerosol mass mixing ratios in the clear
    //   subarea are assumed to be the same as their values before cloud
    //   chemistry (because by definition, cloud chemistry did not happen in
    //   clear sky), while the mixing ratios in the cloudy subarea likely have
    //   changed.
    //----------------------------------------------------------------------------------------
    //   Gases in the clear subarea remain the same as their values before cloud
    //   chemistry.
    //  Here we populate qsub3 for index 1 only as jclea is 1.
    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      if(cnst_is_gas[icnst]) {
        qsub3[icnst][jclea] = qsub2[icnst][jclea];
      }
    }

    // Calculate the gas mixing ratios in the cloudy subarea using the
    // grid-cell mean, cloud fraction and the clear-sky values
    // Here we populate qsub3 for index 2 (jcldy) and adjust index 1 (jclea) if
    // needed.
    compute_qsub_from_gcm_and_qsub_of_other_subarea(kk, cnst_is_gas, fclea,
                                                    fcldy,  // in
                                                    qgcm3, jclea,
                                                    jcldy,          // in
                                                    qsub3, qsub3);  // inout
    if(kk == 48) printf("qsub3_1:%0.15e, %0.15e\n", qsub3[2][1], qsub3[2][2]);
    // qsub3 is fully populated for gasses
    //=========================================================================
    // Set AEROSOL mixing ratios in subareas.
    // Only need to do this for points 2 and 3 in the time integraion loop,
    // i.e., the before-cloud-chem and after-cloud-chem states.
    //=========================================================================
    // Cloud-borne aerosols. (They are straightforward to partition,
    // as they only exist in the cloudy subarea.)
    //----------------------------------------------------------------------------------------
    // Partition mass and number before cloud chemistry
    // NOTE that in this case jclea is 1 and jcldy is 2
    // Following 2 calls set qqcwsub2(:,1)=0 and qqcwsub2(:,2) to a computed
    // value
    set_subarea_qnumb_for_cldbrn_aerosols(jclea, jcldy, fcldy,
                                          qqcwgcm2,   // in
                                          qqcwsub2);  // inout
    // if(kk==48)printf("qqcwsub2_2:%0.15e",qqcwsub2[0][0]);

    set_subarea_qmass_for_cldbrn_aerosols(jclea, jcldy, fcldy,
                                          qqcwgcm2,   // in
                                          qqcwsub2);  // inout
    // if(kk==48)printf("qqcwsub2_3:%0.15e",qqcwsub2[0][0]);
    //  Partition mass and number before cloud chemistry
    // Following 2 calls set qqcwsub3(:,1)=0 and qqcwsub3(:,2) to a computed
    // value
    set_subarea_qnumb_for_cldbrn_aerosols(jclea, jcldy, fcldy,
                                          qqcwgcm3,   // in
                                          qqcwsub3);  // inout
    set_subarea_qmass_for_cldbrn_aerosols(jclea, jcldy, fcldy,
                                          qqcwgcm3,   // in
                                          qqcwsub3);  // inout

    //----------------------------------------------------------------------------------------
    // Interstitial aerosols. (They can exist in both cloudy and clear
    // subareas, and hence need to be partitioned.)
    //----------------------------------------------------------------------------------------
    // Partition mass and number before cloud chemistry
    // Following 2 calls set qsub2(:,1) = 0 and qsub2(:,2) to a computed value
    set_subarea_qnumb_for_intrst_aerosols(jclea, jcldy, fclea, fcldy,  // in
                                          qgcm2, qqcwgcm2, qgcm2,      // in
                                          qsub2);                      // inout

    set_subarea_qmass_for_intrst_aerosols(jclea, jcldy, fclea, fcldy,  // in
                                          qgcm2, qqcwgcm2, qgcm2,      // in
                                          qsub2);                      // inout

    // Partition mass and number before cloud chemistry
    // Following 2 calls set qsub3(:,1) = 0 and qsub3(:,2) to a computed value
    set_subarea_qnumb_for_intrst_aerosols(jclea, jcldy, fclea, fcldy,  // in
                                          qgcm2, qqcwgcm2, qgcm3,      // in
                                          qsub3);                      // inout

    set_subarea_qmass_for_intrst_aerosols(jclea, jcldy, fclea, fcldy,  // in
                                          qgcm2, qqcwgcm2, qgcm3,      // in
                                          qsub3);                      // inout

  }  // different categories
}  // set_subarea_gases_and_aerosols

KOKKOS_INLINE_FUNCTION
void mam_amicphys_1subarea_clear(
    const AmicPhysConfig &config, const int nstep, const Real deltat,
    const int jsub, const int nsubarea, const bool iscldy_subarea,
    const Real afracsub, const Real temp, const Real pmid, const Real pdel,
    const Real zmid, const Real pblh, const Real relhum,
    Real dgn_a[AeroConfig::num_modes()], Real dgn_awet[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()],
    const Real qgas1[AeroConfig::num_gas_ids()],
    const Real qgas3[AeroConfig::num_gas_ids()],
    Real qgas4[AeroConfig::num_gas_ids()],
    Real qgas_delaa[AeroConfig::num_gas_ids()][nqtendaa()],
    const Real qnum3[AeroConfig::num_modes()],
    Real qnum4[AeroConfig::num_modes()],
    Real qnum_delaa[AeroConfig::num_modes()][nqtendaa()],
    const Real qaer3[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer4[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_delaa[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
                   [nqtendaa()],
    const Real qwtr3[AeroConfig::num_modes()],
    Real qwtr4[AeroConfig::num_modes()]) {
  static constexpr int num_gas_ids     = AeroConfig::num_gas_ids();
  static constexpr int num_modes       = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();

  static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  // Turn off nh3 for now.  This is a future enhancement.
  static constexpr int igas_nh3 = -999888777;  // Same as mam_refactor
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_pom = static_cast<int>(AeroId::POM);

  const AeroId gas_to_aer[num_gas_ids] = {AeroId::SOA, AeroId::SO4,
                                          AeroId::None};

  const bool l_gas_condense_to_mode[num_gas_ids][num_modes] = {
      {true, true, true, true},
      {true, true, true, true},
      {false, false, false, false}};
  enum { NA, ANAL, IMPL };
  const int eqn_and_numerics_category[num_gas_ids] = {IMPL, ANAL, ANAL};

  // air molar density (kmol/m3)
  // BAD CONSTANT
  const Real r_universal = 8.314467591;  // [mJ/(mol)] as in mam_refactor
  const Real aircon      = pmid / (1000 * r_universal * temp);
  const Real alnsg_aer[num_modes] = {0.58778666490211906, 0.47000362924573563,
                                     0.58778666490211906, 0.47000362924573563};
  const Real uptk_rate_factor[num_gas_ids] = {0.81, 1.0, 1.0};
  // calculates changes to gas and aerosol sub-area TMRs (tracer mixing
  // ratios) qgas3, qaer3, qnum3 are the current incoming TMRs qgas4, qaer4,
  // qnum4 are the updated outgoing TMRs
  //
  // this routine calculates changes involving
  //    gas-aerosol exchange (condensation/evaporation)
  //    growth from smaller to larger modes (renaming) due to condensation
  //    new particle nucleation
  //    coagulation
  //    transfer of particles from hydrophobic modes to hydrophilic modes
  //    (aging)
  //       due to condensation and coagulation
  //
  // qXXXN (X=gas,aer,wat,num; N=1:4) are sub-area mixing ratios
  //    XXX=gas - gas species
  //    XXX=aer - aerosol mass  species (excluding water)
  //    XXX=wat - aerosol water
  //    XXX=num - aerosol number
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - current incoming values (before gas-aerosol exchange, newnuc,
  //    coag) N=4 - updated outgoing values (after  gas-aerosol exchange,
  //    newnuc, coag)
  //
  // qXXX_delaa are TMR changes (not tendencies)
  //    for different processes, which are used to produce history output
  // for a clear sub-area, the processes are condensation/evaporation (and
  // associated aging), renaming, coagulation, and nucleation

  Real qgas_cur[num_gas_ids];
  for(int i = 0; i < num_gas_ids; ++i) qgas_cur[i] = qgas3[i];
  Real qaer_cur[num_aerosol_ids][num_modes];
  for(int i = 0; i < num_aerosol_ids; ++i)
    for(int j = 0; j < num_modes; ++j) qaer_cur[i][j] = qaer3[i][j];

  Real qnum_cur[num_modes];
  for(int j = 0; j < num_modes; ++j) qnum_cur[j] = qnum3[j];
  Real qwtr_cur[num_modes];
  for(int j = 0; j < num_modes; ++j) qwtr_cur[j] = qwtr3[j];

  // qgas_netprod_otrproc = gas net production rate from other processes
  //    such as gas-phase chemistry and emissions (mol/mol/s)
  // this allows the condensation (gasaerexch) routine to apply production and
  // condensation loss
  //    together, which is more accurate numerically
  // NOTE - must be >= zero, as numerical method can fail when it is negative
  // NOTE - currently only the values for h2so4 and nh3 should be non-zero
  Real qgas_netprod_otrproc[num_gas_ids] = {};
  if(config.do_cond && config.gaexch_h2so4_uptake_optaa == 2) {
    for(int igas = 0; igas < num_gas_ids; ++igas) {
      if(igas == igas_h2so4 || igas == igas_nh3) {
        // if config.gaexch_h2so4_uptake_optaa == 2, then
        //    if qgas increases from pre-gaschem to post-cldchem,
        //       start from the pre-gaschem mix-ratio and add in the
        //       production during the integration
        //    if it decreases,
        //       start from post-cldchem mix-ratio
        // *** currently just do this for h2so4 and nh3
        qgas_netprod_otrproc[igas] = (qgas3[igas] - qgas1[igas]) / deltat;
        if(qgas_netprod_otrproc[igas] >= 0.0)
          qgas_cur[igas] = qgas1[igas];
        else
          qgas_netprod_otrproc[igas] = 0.0;
      }
    }
  }
  Real qgas_del_cond[num_gas_ids]                                      = {};
  Real qgas_del_nnuc[num_gas_ids]                                      = {};
  Real qgas_del_cond_only[num_gas_ids]                                 = {};
  Real qaer_del_cond[num_aerosol_ids][num_modes]                       = {};
  Real qaer_del_rnam[num_aerosol_ids][num_modes]                       = {};
  Real qaer_del_nnuc[num_aerosol_ids][num_modes]                       = {};
  Real qaer_del_coag[num_aerosol_ids][num_modes]                       = {};
  Real qaer_delsub_coag_in[num_aerosol_ids][AeroConfig::max_agepair()] = {};
  Real qaer_delsub_cond[num_aerosol_ids][num_modes]                    = {};
  Real qaer_delsub_coag[num_aerosol_ids][num_modes]                    = {};
  Real qaer_del_cond_only[num_aerosol_ids][num_modes]                  = {};
  Real qnum_del_cond[num_modes]                                        = {};
  Real qnum_del_rnam[num_modes]                                        = {};
  Real qnum_del_nnuc[num_modes]                                        = {};
  Real qnum_del_coag[num_modes]                                        = {};
  Real qnum_delsub_cond[num_modes]                                     = {};
  Real qnum_delsub_coag[num_modes]                                     = {};
  Real qnum_del_cond_only[num_modes]                                   = {};
  Real dnclusterdt                                                     = 0.0;

  const int ntsubstep = 1;
  Real dtsubstep      = deltat;
  if(ntsubstep > 1) dtsubstep = deltat / ntsubstep;
  Real del_h2so4_gasprod =
      haero::max(qgas3[igas_h2so4] - qgas1[igas_h2so4], 0.0) / ntsubstep;

  // loop over multiple time sub-steps
  for(int jtsubstep = 1; jtsubstep <= ntsubstep; ++jtsubstep) {
    // gas-aerosol exchange
    Real uptkrate_h2so4                                    = 0.0;
    Real del_h2so4_aeruptk                                 = 0.0;
    Real qaer_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};
    Real qgas_avg[num_gas_ids]                             = {};
    Real qnum_sv1[num_modes]                               = {};
    Real qaer_sv1[num_aerosol_ids][num_modes]              = {};
    Real qgas_sv1[num_gas_ids]                             = {};

    if(config.do_cond) {
      const bool l_calc_gas_uptake_coeff   = jtsubstep == 1;
      Real uptkaer[num_gas_ids][num_modes] = {};

      for(int i = 0; i < num_gas_ids; ++i) qgas_sv1[i] = qgas_cur[i];
      for(int i = 0; i < num_modes; ++i) qnum_sv1[i] = qnum_cur[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) qaer_sv1[j][i] = qaer_cur[j][i];

      // time sub-step
      const Real dtsub_soa_fixed = -1.0;
      // Integration order
      const int nghq         = 2;
      const int ntot_soamode = 4;
      int niter_out          = 0;
      Real g0_soa_out        = 0;
      gasaerexch::mam_gasaerexch_1subarea(
          nghq, igas_h2so4, igas_nh3, ntot_soamode, gas_to_aer, iaer_so4,
          iaer_pom, l_calc_gas_uptake_coeff, l_gas_condense_to_mode,
          eqn_and_numerics_category, dtsubstep, dtsub_soa_fixed, temp, pmid,
          aircon, num_gas_ids, qgas_cur, qgas_avg, qgas_netprod_otrproc,
          qaer_cur, qnum_cur, dgn_awet, alnsg_aer, uptk_rate_factor, uptkaer,
          uptkrate_h2so4, niter_out, g0_soa_out);

      if(config.newnuc_h2so4_conc_optaa == 11)
        qgas_avg[igas_h2so4] =
            0.5 * (qgas_sv1[igas_h2so4] + qgas_cur[igas_h2so4]);
      else if(config.newnuc_h2so4_conc_optaa == 12)
        qgas_avg[igas_h2so4] = qgas_cur[igas_h2so4];

      for(int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond[i] +=
            (qgas_cur[i] - (qgas_sv1[i] + qgas_netprod_otrproc[i] * dtsubstep));

      for(int i = 0; i < num_modes; ++i)
        qnum_delsub_cond[i] = qnum_cur[i] - qnum_sv1[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_delsub_cond[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];

      // qaer_del_grow4rnam = change in qaer_del_cond during latest
      // condensation calculations
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_delsub_grow4rnam[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];
      for(int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond_only[i] = qgas_del_cond[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_del_cond_only[i][j] = qaer_delsub_cond[i][j];
      for(int i = 0; i < num_modes; ++i)
        qnum_del_cond_only[i] = qnum_delsub_cond[i];
      del_h2so4_aeruptk =
          qgas_cur[igas_h2so4] -
          (qgas_sv1[igas_h2so4] + qgas_netprod_otrproc[igas_h2so4] * dtsubstep);
    } else {
      for(int i = 0; i < num_gas_ids; ++i) qgas_avg[i] = qgas_cur[i];
    }

    // renaming after "continuous growth"
    if(config.do_rename) {
      constexpr int nmodes                = AeroConfig::num_modes();
      constexpr int naerosol_species      = AeroConfig::num_aerosol_ids();
      const Real smallest_dryvol_value    = 1.0e-25;  // BAD_CONSTANT
      const int dest_mode_of_mode[nmodes] = {-1, 0, -1, -1};

      Real qnumcw_cur[num_modes]                               = {};
      Real qaercw_cur[num_aerosol_ids][num_modes]              = {};
      Real qaercw_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};
      Real mean_std_dev[nmodes];
      Real fmode_dist_tail_fac[nmodes];
      Real v2n_lo_rlx[nmodes];
      Real v2n_hi_rlx[nmodes];
      Real ln_diameter_tail_fac[nmodes];
      int num_pairs = 0;
      Real diameter_cutoff[nmodes];
      Real ln_dia_cutoff[nmodes];
      Real diameter_threshold[nmodes];
      Real mass_2_vol[naerosol_species] = {0.15,
                                           6.4971751412429377e-002,
                                           0.15,
                                           7.0588235294117650e-003,
                                           3.0789473684210526e-002,
                                           5.1923076923076926e-002,
                                           156.20986883198000};

      rename::find_renaming_pairs(dest_mode_of_mode,     // in
                                  mean_std_dev,          // out
                                  fmode_dist_tail_fac,   // out
                                  v2n_lo_rlx,            // out
                                  v2n_hi_rlx,            // out
                                  ln_diameter_tail_fac,  // out
                                  num_pairs,             // out
                                  diameter_cutoff,       // out
                                  ln_dia_cutoff, diameter_threshold);

      for(int i = 0; i < num_modes; ++i) qnum_sv1[i] = qnum_cur[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) qaer_sv1[j][i] = qaer_cur[j][i];
      Real dgnum_amode[nmodes];
      for(int m = 0; m < nmodes; ++m) {
        dgnum_amode[m] = modes(m).nom_diameter;
      }

      {
        Real qmol_i_cur[num_modes][num_aerosol_ids];
        Real qmol_i_del[num_modes][num_aerosol_ids];
        Real qmol_c_cur[num_modes][num_aerosol_ids];
        Real qmol_c_del[num_modes][num_aerosol_ids];
        for(int j = 0; j < num_aerosol_ids; ++j)
          for(int i = 0; i < num_modes; ++i) {
            qmol_i_cur[i][j] = qaer_cur[j][i];
            qmol_i_del[i][j] = qaer_delsub_grow4rnam[j][i];
            qmol_c_cur[i][j] = qaercw_cur[j][i];
            qmol_c_del[i][j] = qaercw_delsub_grow4rnam[j][i];
          }
        Rename rename;
        rename.mam_rename_1subarea_(
            iscldy_subarea, smallest_dryvol_value, dest_mode_of_mode,
            mean_std_dev, fmode_dist_tail_fac, v2n_lo_rlx, v2n_hi_rlx,
            ln_diameter_tail_fac, num_pairs, diameter_cutoff, ln_dia_cutoff,
            diameter_threshold, mass_2_vol, dgnum_amode, qnum_cur, qmol_i_cur,
            qmol_i_del, qnumcw_cur, qmol_c_cur, qmol_c_del);

        for(int j = 0; j < num_aerosol_ids; ++j)
          for(int i = 0; i < num_modes; ++i) {
            qaer_cur[j][i]                = qmol_i_cur[i][j];
            qaer_delsub_grow4rnam[j][i]   = qmol_i_del[i][j];
            qaercw_cur[j][i]              = qmol_c_cur[i][j];
            qaercw_delsub_grow4rnam[j][i] = qmol_c_del[i][j];
          }
      }

      for(int i = 0; i < num_modes; ++i)
        qnum_del_rnam[i] += qnum_cur[i] - qnum_sv1[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_del_rnam[i][j] += qaer_cur[i][j] - qaer_sv1[i][j];
    }

    // new particle formation (nucleation)
    if(config.do_newnuc) {
      for(int i = 0; i < num_gas_ids; ++i) qgas_sv1[i] = qgas_cur[i];
      for(int i = 0; i < num_modes; ++i) qnum_sv1[i] = qnum_cur[i];
      Real qaer_cur_tmp[num_modes][num_aerosol_ids];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) {
          qaer_sv1[j][i]     = qaer_cur[j][i];
          qaer_cur_tmp[i][j] = qaer_cur[j][i];
        }
      Real dnclusterdt_substep = 0;
      Real dndt_ait            = 0;
      Real dmdt_ait            = 0;
      Real dso4dt_ait          = 0;
      Real dnh4dt_ait          = 0;
      Nucleation nucleation;
      Nucleation::Config config;
      config.dens_so4a_host   = 1770;
      config.mw_nh4a_host     = 115;
      config.mw_so4a_host     = 115;
      config.accom_coef_h2so4 = 0.65;
      AeroConfig aero_config;
      nucleation.init(aero_config, config);
      nucleation.compute_tendencies_(
          dtsubstep, temp, pmid, aircon, zmid, pblh, relhum, uptkrate_h2so4,
          del_h2so4_gasprod, del_h2so4_aeruptk, qgas_cur, qgas_avg, qnum_cur,
          qaer_cur_tmp, qwtr_cur, dndt_ait, dmdt_ait, dso4dt_ait, dnh4dt_ait,
          dnclusterdt_substep);
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) qaer_cur[j][i] = qaer_cur_tmp[i][j];

      //! Apply the tendencies to the prognostics.
      const int nait = static_cast<int>(ModeIndex::Aitken);
      qnum_cur[nait] += dndt_ait * dtsubstep;

      if(dso4dt_ait > 0.0) {
        static constexpr int iaer_so4   = static_cast<int>(AeroId::SO4);
        static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);

        Real delta_q = dso4dt_ait * dtsubstep;
        qaer_cur[iaer_so4][nait] += delta_q;
        delta_q = haero::min(delta_q, qgas_cur[igas_h2so4]);
        qgas_cur[igas_h2so4] -= delta_q;
      }

      if(igas_nh3 > 0 && dnh4dt_ait > 0.0) {
        static constexpr int iaer_nh4 =
            -9999999;  // static_cast<int>(AeroId::NH4);

        Real delta_q = dnh4dt_ait * dtsubstep;
        qaer_cur[iaer_nh4][nait] += delta_q;
        delta_q = haero::min(delta_q, qgas_cur[igas_nh3]);
        qgas_cur[igas_nh3] -= delta_q;
      }
      for(int i = 0; i < num_gas_ids; ++i)
        qgas_del_nnuc[i] += (qgas_cur[i] - qgas_sv1[i]);
      for(int i = 0; i < num_modes; ++i)
        qnum_del_nnuc[i] += (qnum_cur[i] - qnum_sv1[i]);
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i)
          qaer_del_nnuc[j][i] += (qaer_cur[j][i] - qaer_sv1[j][i]);

      dnclusterdt = dnclusterdt + dnclusterdt_substep * (dtsubstep / deltat);
    }

    // coagulation part
    if(config.do_coag) {
      for(int i = 0; i < num_modes; ++i) qnum_sv1[i] = qnum_cur[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) qaer_sv1[j][i] = qaer_cur[j][i];
      coagulation::mam_coag_1subarea(dtsubstep, temp, pmid, aircon, dgn_a,
                                     dgn_awet, wetdens, qnum_cur, qaer_cur,
                                     qaer_delsub_coag_in);
      for(int i = 0; i < num_modes; ++i)
        qnum_delsub_coag[i] = qnum_cur[i] - qnum_sv1[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i)
          qaer_delsub_coag[j][i] = qaer_cur[j][i] - qaer_sv1[j][i];
    }

    // primary carbon aging

    aging::mam_pcarbon_aging_1subarea(
        dgn_a, qnum_cur, qnum_delsub_cond, qnum_delsub_coag, qaer_cur,
        qaer_delsub_cond, qaer_delsub_coag, qaer_delsub_coag_in);

    // accumulate sub-step q-dels
    if(config.do_coag) {
      for(int i = 0; i < num_modes; ++i)
        qnum_del_coag[i] += qnum_delsub_coag[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i)
          qaer_del_coag[j][i] += qaer_delsub_coag[j][i];
    }
    if(config.do_cond) {
      for(int i = 0; i < num_modes; ++i)
        qnum_del_cond[i] += qnum_delsub_cond[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i)
          qaer_del_cond[j][i] += qaer_delsub_cond[j][i];
    }
  }

  // final mix ratios
  for(int i = 0; i < num_gas_ids; ++i) qgas4[i] = qgas_cur[i];
  for(int j = 0; j < num_aerosol_ids; ++j)
    for(int i = 0; i < num_modes; ++i) qaer4[j][i] = qaer_cur[j][i];
  for(int i = 0; i < num_modes; ++i) qnum4[i] = qnum_cur[i];
  for(int i = 0; i < num_modes; ++i) qwtr4[i] = qwtr_cur[i];

  // final mix ratio changes
  for(int i = 0; i < num_gas_ids; ++i) {
    qgas_delaa[i][iqtend_cond()]      = qgas_del_cond[i];
    qgas_delaa[i][iqtend_rnam()]      = 0.0;
    qgas_delaa[i][iqtend_nnuc()]      = qgas_del_nnuc[i];
    qgas_delaa[i][iqtend_coag()]      = 0.0;
    qgas_delaa[i][iqtend_cond_only()] = qgas_del_cond_only[i];
  }
  for(int i = 0; i < num_modes; ++i) {
    qnum_delaa[i][iqtend_cond()]      = qnum_del_cond[i];
    qnum_delaa[i][iqtend_rnam()]      = qnum_del_rnam[i];
    qnum_delaa[i][iqtend_nnuc()]      = qnum_del_nnuc[i];
    qnum_delaa[i][iqtend_coag()]      = qnum_del_coag[i];
    qnum_delaa[i][iqtend_cond_only()] = qnum_del_cond_only[i];
  }
  for(int j = 0; j < num_aerosol_ids; ++j) {
    for(int i = 0; i < num_modes; ++i) {
      qaer_delaa[j][i][iqtend_cond()]      = qaer_del_cond[j][i];
      qaer_delaa[j][i][iqtend_rnam()]      = qaer_del_rnam[j][i];
      qaer_delaa[j][i][iqtend_nnuc()]      = qaer_del_nnuc[j][i];
      qaer_delaa[j][i][iqtend_coag()]      = qaer_del_coag[j][i];
      qaer_delaa[j][i][iqtend_cond_only()] = qaer_del_cond_only[j][i];
    }
  }
}
//--------------------------------------------------------------------------------
// Utility functions
//--------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void copy_1d_array(const int arr_len, const Real (&arr_in)[arr_len],  // in
                   Real (&arr_out)[arr_len]) {                        // out
  for(int ii = 0; ii < arr_len; ++ii) {
    arr_out[ii] = arr_in[ii];
  }
}

KOKKOS_INLINE_FUNCTION
void copy_2d_array(const int first_dimlen,                             // in
                   const int second_dimlen,                            // in
                   const Real (&arr_in)[first_dimlen][second_dimlen],  // in
                   Real (&arr_out)[first_dimlen][second_dimlen]) {     // out

  for(int ifd = 0; ifd < first_dimlen; ++ifd) {
    for(int isd = 0; isd < second_dimlen; ++isd) {
      arr_out[ifd][isd] = arr_in[ifd][isd];
    }
  }
}
template <typename DT>
KOKKOS_INLINE_FUNCTION void assign_1d_array(const int arr_len,
                                            const DT num,   // in
                                            DT *arr_out) {  // out
  for(int ii = 0; ii < arr_len; ++ii) {
    arr_out[ii] = num;
  }
}

KOKKOS_INLINE_FUNCTION
void assign_2d_array(const int first_dimlen,                          // in
                     const int second_dimlen,                         // in
                     const Real num,                                  // in
                     Real (&arr_out)[first_dimlen][second_dimlen]) {  // out
  for(int ifd = 0; ifd < first_dimlen; ++ifd) {
    for(int isd = 0; isd < second_dimlen; ++isd) {
      arr_out[ifd][isd] = num;
    }
  }
}
// copy 3d arrays
KOKKOS_INLINE_FUNCTION
void assign_3d_array(
    const int first_dimlen,                                        // in
    const int second_dimlen,                                       // in
    const int third_dimlen,                                        // in
    const Real num,                                                // in
    Real (&arr_out)[first_dimlen][second_dimlen][third_dimlen]) {  // out
  for(int ifd = 0; ifd < first_dimlen; ++ifd) {
    for(int isd = 0; isd < second_dimlen; ++isd) {
      for(int itd = 0; itd < third_dimlen; ++itd) {
        arr_out[ifd][isd][itd] = num;
      }
    }
  }
}

//--------------------------------------------------------------------------------
// Call aerosol microphysics processes for a single (cloudy or clear) subarea
// (with indices = lchnk,ii,kk,jsubarea)
//
// qgas3, qaer3, qaercw3, qnum3, qnumcw3 are the current incoming TMRs
// qgas_cur, qaer_cur, qaercw_cur, qnum_cur, qnumcw_cur are the updated
// outgoing TMRs
//
// In a clear subarea, calculate
//  - gas-aerosol exchange (condensation/evaporation)
//  - growth from smaller to larger modes (renaming) due to condensation
//  - new particle nucleation
//  - coagulation
//  - transfer of particles from hydrophobic modes to hydrophilic modes
//  (aging)
//    due to condensation and coagulation
//
// In a cloudy subarea,
//  - when do_cond = false, this routine only calculate changes involving
//    growth from smaller to larger modes (renaming) following cloud chemistry
//    so gas TMRs are not changed
//  - when do_cond = true, this routine also calculates changes involving
//    gas-aerosol exchange (condensation/evaporation)
//  - transfer of particles from hydrophobic modes to hydrophilic modes
//  (aging)
//       due to condensation
// Currently, in a cloudy subarea, this routine does not do
//  - new particle nucleation - because h2so4 gas conc. should be very low in
//  cloudy air
//  - coagulation - because cloud-borne aerosol would need to be included
//--------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void mam_amicphys_1subarea(
    // in
    const Real gaexch_h2so4_uptake_optaa, const bool do_cond_sub,
    const bool do_rename_sub, const bool do_newnuc_sub, const bool do_coag_sub,
    const int ii, const int kk, const Real deltat, const int jsubarea,
    const int nsubarea, const bool iscldy_subarea, const Real afracsub,
    const Real temp, const Real pmid, const Real pdel, const Real zmid,
    const Real pblh, const Real relhumsub,
    const Real (&dgn_a)[AeroConfig::num_modes()],
    const Real (&dgn_awet)[AeroConfig::num_modes()],
    const Real (&wetdens)[AeroConfig::num_modes()],
    const Real (&qgas1)[max_gas()], const Real (&qgas3)[max_gas()],
    // inout
    Real (&qgas_cur)[max_gas()], Real (&qgas_delaa)[max_gas()][nqtendaa()],
    // in
    const Real (&qnum3)[AeroConfig::num_modes()],
    // inout
    Real (&qnum_cur)[AeroConfig::num_modes()],
    Real (&qnum_delaa)[AeroConfig::num_modes()][nqtendaa()],
    // in
    const Real (&qaer2)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    const Real (&qaer3)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    // inout
    Real (&qaer_cur)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real (&qaer_delaa)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
                      [nqtendaa()],
    // in
    const Real (&qwtr3)[AeroConfig::num_modes()],
    // inout
    Real (&qwtr_cur)[AeroConfig::num_modes()],
    // in
    const Real (&qnumcw3)[AeroConfig::num_modes()],
    // inout
    Real (&qnumcw_cur)[AeroConfig::num_modes()],
    Real (&qnumcw_delaa)[AeroConfig::num_modes()][nqqcwtendaa()],
    // in
    const Real (
        &qaercw2)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    const Real (
        &qaercw3)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    // inout
    Real (&qaercw_cur)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real (&qaercw_delaa)[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
                        [nqqcwtendaa()])

{
  // do_cond_sub, do_rename_sub: true if the aero. microp. process is
  //                             calculated in this subarea
  // do_newnuc_sub, do_coag_sub: true if the aero.  microp. process is
  //                             calculated in this subarea
  // iscldy_subarea: true if sub-area is cloudy
  // ii, kk: column and level indices
  // jsubarea, nsubarea: sub-area index, number of sub-areas
  // afracsub: fractional area of subarea [unitless, 0-1]
  // deltat: time step [s]
  // temp: air temperature at model levels [K]
  // pmid: air pressure at layer center [Pa]
  // pdel: pressure thickness of layer [Pa]
  // zmid: altitude (above ground) at layer center [m]
  // pblh: planetary boundary layer depth [m]
  // relhum: relative humidity [unitless, 0-1]
  // dgn_a   (max_mode): dry geo. mean diameter [m] of number distribution
  // dgn_awet(max_mode): wet geo. mean diameter [m] of number distribution
  // wetdens (max_mode): interstitial aerosol wet density [kg/m3]

  // Subare mixing ratios qXXXN (X=gas,aer,wat,num; N=1:4):
  //
  //    XXX=gas - gas species [kmol/kmol]
  //    XXX=aer - aerosol mass species (excluding water) [kmol/kmol]
  //    XXX=wat - aerosol water [kmol/kmol]
  //    XXX=num - aerosol number [#/kmol]
  //
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - current incoming values (before gas-aerosol exchange, newnuc,
  //    coag) N=_cur - updated outgoing values (after  gas-aerosol exchange,
  //    newnuc, coag)
  //
  // qgas1, qgas3 [kmol/kmol]
  // qgas_cur     [kmol/kmol]

  // qnum3    [#/kmol]
  // qnum_cur [#/kmol]

  // qaer2, qaer3 [kmol/kmol]
  // qaer_cur[kmol/kmol]

  // qnumcw3[#/kmol]
  // qnumcw_cur  [#/kmol]

  // qaercw2, qaercw3 [kmol/kmol]
  // qaercw_cur  [kmol/kmol]

  // qwtr3      [kmol/kmol]
  // qwtr_cur   [kmol/kmol]

  // qXXX_delaa are TMR changes (increments, not tendencies) of different
  // microphysics processes. These are diagnostics sent to history output;
  // they do not directly affect time integration.

  // qgas_delaa   [kmol/kmol]
  // qnum_delaa   [   #/kmol]
  // qaer_delaa   [kmol/kmol]
  // qnumcw_delaa [   #/kmol]
  // qaercw_delaa [kmol/kmol]

  // type ( misc_vars_aa_type ), intent(inout) :: misc_vars_aa_sub

  //---------------------------------------------------------------------------------------
  // Calculate air molar density [kmol/m3] to be passed on to individual
  // parameterizations
  //---------------------------------------------------------------------------------------
  // BAD CONSTANT
  // Universal gas constant (J/K/kmol)
  constexpr Real r_universal = 8314.46759100000;
  const Real aircon          = pmid / (r_universal * temp);

  //----------------------------------------------------------
  // Initializ mixing ratios with the before-amicphys values
  //----------------------------------------------------------

  copy_1d_array(max_gas(), qgas3,  // in
                qgas_cur);         // out

  if(kk == 48)
    printf("qgas_cur1:%0.15e,%0.15e,%0.15e\n", qgas_cur[0], qgas_cur[1],
           aircon);
  constexpr int nmodes   = AeroConfig::num_modes();
  constexpr int nspecies = AeroConfig::num_aerosol_ids();

  copy_2d_array(nspecies, nmodes, qaer3,  // in
                qaer_cur);                // out

  copy_1d_array(nmodes, qnum3,  // in
                qnum_cur);      // out

  copy_1d_array(nmodes, qwtr3,  // in
                qwtr_cur);      // out

  if(iscldy_subarea) {
    copy_1d_array(nmodes, qnumcw3,            // in
                  qnumcw_cur);                // out
    copy_2d_array(nspecies, nmodes, qaercw3,  // in
                  qaercw_cur);                // out
  }                                           // iscldy_subarea

  if(kk == 48) {
    for(int imode = 0; imode < nmodes; ++imode) {
      printf("qnum_cur:%0.15e,%0.15e,%0.15e, %i\n", qnum_cur[imode],
             qwtr_cur[imode], qnumcw_cur[imode], imode);
      for(int iaer = 0; iaer < nspecies; ++iaer) {
        printf("qaer_cur:%0.15e,%0.15e, %i, %i\n", qaer_cur[iaer][imode],
               qaercw_cur[iaer][imode], iaer, imode);
      }
    }
  }

  //---------------------------------------------------------------------
  // Diagnose net production rate of H2SO4 gas production
  // cause by other processes (e.g., gas chemistry and cloud chemistry)
  //---------------------------------------------------------------------
  Real qgas_netprod_otrproc[max_gas()] = {0};

  // If gaexch_h2so4_uptake_optaa == 2, then
  //  - if qgas increases from pre-gaschem to post-cldchem,
  //    start from the pre-gaschem mix-ratio and add in the production during
  //    the integration
  //  - if it decreases,  start from post-cldchem mix-ratio
  // *** currently just do this for h2so4 (and nh3 if considered in model)
  constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  constexpr int igas_nh3   = -999888777;  // Same as mam_refactor

  if((do_cond_sub) && (gaexch_h2so4_uptake_optaa == 2)) {
    for(int igas = 0; igas < max_gas(); ++igas) {
      if((igas == igas_h2so4) || (igas == igas_nh3)) {
        qgas_netprod_otrproc[igas] = (qgas3[igas] - qgas1[igas]) / deltat;
        qgas_cur[igas] = (qgas_netprod_otrproc[igas] >= 0) ? qgas1[igas] : 0;
      }  // h2so4, igas_nh3
    }    // igas
  }      // do_cond_sub,gaexch_h2so4_uptake_optaa

  constexpr int ntsubstep = 1;
  const Real del_h2so4_gasprod =
      haero::max(qgas3[igas_h2so4] - qgas1[igas_h2so4], 0) / ntsubstep;

  //-----------------------------------
  // Initialize increment diagnostics
  //-----------------------------------

  assign_2d_array(max_gas(), nqtendaa(), 0,  // in
                  qgas_delaa);               // out

  assign_2d_array(nmodes, nqtendaa(), 0,  // in
                  qnum_delaa);            // out

  assign_3d_array(nspecies, nmodes, nqtendaa(), 0,  // in
                  qaer_delaa);                      // out

  assign_2d_array(nmodes, nqqcwtendaa(), 0,  // in
                  qnumcw_delaa);             // out

  assign_3d_array(nspecies, nmodes, nqqcwtendaa(), 0,  // in
                  qaercw_delaa);                       // out

  Real ncluster_tend_nnuc_1grid = 0;

  //***********************************
  // loop over multiple time sub-steps
  //***********************************
  // FIXME: add assert statement for ntsubstep
  const int dtsubstep = deltat / ntsubstep;

  Real qgas_sv1[max_gas()];
  Real qnum_sv1[nmodes];
  Real qaer_sv1[nspecies][nmodes];

  Real del_h2so4_aeruptk;    // [kmol/kmol]
  Real qgas_avg[max_gas()];  // [kmol/kmol]

  // Mixing ratio increments of sub-timesteps used for process coupling

  Real qnum_delsub_cond[nmodes];                   // [   #/kmol]
  Real qnum_delsub_coag[nmodes];                   // [   #/kmol]
  Real qaer_delsub_cond[nspecies][nmodes];         // [   #/kmol]
  Real qaer_delsub_coag[nspecies][nmodes];         // [kmol/kmol]
  Real qaer_delsub_grow4rnam[nspecies][nmodes];    // [kmol/kmol]
  Real qaercw_delsub_grow4rnam[nspecies][nmodes];  // [kmol/kmol]

  constexpr int max_agepair = AeroConfig::max_agepair();
  Real qaer_delsub_coag_in[nspecies][max_agepair];  // [kmol/kmol]
  // FIXME: an aeert statement for ntsubstep
  for(int jtsubstep = 0; jtsubstep < ntsubstep; ++jtsubstep) {
    //======================
    // Gas-aerosol exchange
    //======================
    Real uptkrate_h2so4 = 0;

    if(do_cond_sub) {
      copy_1d_array(max_gas(), qgas_cur,  // in
                    qgas_sv1);            // out
      copy_1d_array(nmodes, qnum_cur,     // in
                    qnum_sv1);            // out

      copy_2d_array(nspecies, nmodes, qaer_cur,  // in
                    qaer_sv1);                   // out
#if 0

         call mam_gasaerexch_1subarea(                   
           nstep,             lchnk,                     
           ii,                kk,               jsubarea,
           jtsubstep,         ntsubstep,                 
           latndx,            lonndx,           lund,    
           dtsubstep,                                    
           temp,              pmid,             aircon,  
           n_mode,                                       
           qgas_cur,          qgas_avg,                  
           qgas_netprod_otrproc,                         
           qaer_cur,                                     
           qnum_cur,                                     
           qwtr_cur,                                     
           dgn_a,             dgn_awet,         wetdens, 
           uptkaer,           uptkrate_h2so4              )
           if(print_out .and. ii==icolprnt(lchnk) .and. kk==kprnt) write(106,*)'qgas_cur3:',qgas_cur(1),qgas_cur(2)

         qgas_delaa(:,iqtend_cond) = qgas_delaa(:,iqtend_cond) + (qgas_cur - (qgas_sv1 + qgas_netprod_otrproc*dtsubstep)) 

         qnum_delsub_cond = qnum_cur - qnum_sv1
         qaer_delsub_cond = qaer_cur - qaer_sv1

         del_h2so4_aeruptk = qgas_cur(igas_h2so4)
                           - (qgas_sv1(igas_h2so4) + qgas_netprod_otrproc(igas_h2so4)*dtsubstep)
#endif
    } else {                              // do_cond_sub
      copy_1d_array(max_gas(), qgas_cur,  // in
                    qgas_avg);            // out

      assign_2d_array(nspecies, nmodes, 0,  // in
                      qaer_delsub_cond);    // out

      assign_1d_array(nmodes, 0.0,        // in
                      qnum_delsub_cond);  // out
      del_h2so4_aeruptk = 0;

    }  // do_cond_sub

    //====================================
    // Renaming after "continuous growth"
    //====================================
    if(do_rename_sub) {
      constexpr int dest_mode_of_mode[nmodes] = {-1, 0, -1, -1};

      //---------------------------------------------------------
      // Calculate changes in aerosol mass mixing ratios due to
      //  - gas condensation/evaporation
      //  - cloud chemistry (if the subarea is cloudy)
      //---------------------------------------------------------
      copy_2d_array(nspecies, nmodes, qaer_delsub_cond,  // in
                    qaer_delsub_grow4rnam);              // out

      if(iscldy_subarea) {
        for(int is = 0; is < nspecies; ++is) {
          for(int im = 0; im < nmodes; ++im) {
            qaer_delsub_grow4rnam[is][im] =
                (qaer3[is][im] - qaer2[is][im]) / ntsubstep +
                qaer_delsub_grow4rnam[is][im];
            qaercw_delsub_grow4rnam[is][im] =
                (qaercw3[is][im] - qaercw2[is][im]) / ntsubstep;
          }
        }
      }

      //----------
      // Renaming
      //----------
      copy_1d_array(nmodes, qnum_cur,  // in
                    qnum_sv1);         // out

      copy_2d_array(nspecies, nmodes, qaer_cur,  // in
                    qaer_sv1);                   // out

      Real qnumcw_sv1[nmodes];
      copy_1d_array(nmodes, qnumcw_cur,  // in
                    qnumcw_sv1);         // out
      Real qaercw_sv1[nspecies][nmodes];
      copy_2d_array(nspecies, nmodes, qaercw_cur,  // in
                    qaercw_sv1);                   // out

      Real mean_std_dev[nmodes];
      Real fmode_dist_tail_fac[nmodes];
      Real v2n_lo_rlx[nmodes];
      Real v2n_hi_rlx[nmodes];
      Real ln_diameter_tail_fac[nmodes];
      int num_pairs = 0;
      Real diameter_cutoff[nmodes];
      Real ln_dia_cutoff[nmodes];
      Real diameter_threshold[nmodes];

      rename::find_renaming_pairs(
          dest_mode_of_mode,                              // in
          mean_std_dev, fmode_dist_tail_fac, v2n_lo_rlx,  // out
          v2n_hi_rlx, ln_diameter_tail_fac, num_pairs,    // out
          diameter_cutoff, ln_dia_cutoff,                 // out
          diameter_threshold);                            // out
      constexpr Real mass_2_vol[nspecies] = {0.15,
                                             6.4971751412429377e-002,
                                             0.15,
                                             7.0588235294117650e-003,
                                             3.0789473684210526e-002,
                                             5.1923076923076926e-002,
                                             156.20986883198000};
      Real dgnum_amode[nmodes];
      for(int m = 0; m < nmodes; ++m) {
        dgnum_amode[m] = modes(m).nom_diameter;
      }
      // BAD_CONSTANT
      constexpr Real smallest_dryvol_value = 1.0e-25;

      // swap dimensions as mam_rename_1subarea_ uses output arrays in
      //  a swapped dimension order
      Real qaer_cur_tmp[nmodes][nspecies];
      Real qaer_delsub_grow4rnam_tmp[nmodes][nspecies];
      Real qaercw_cur_tmp[nmodes][nspecies];
      Real qaercw_delsub_grow4rnam_tmp[nmodes][nspecies];
      for(int is = 0; is < nspecies; ++is) {
        for(int im = 0; im < nmodes; ++im) {
          qaer_cur_tmp[im][is]                = qaer_cur[is][im];
          qaer_delsub_grow4rnam_tmp[im][is]   = qaer_delsub_grow4rnam[is][im];
          qaercw_cur_tmp[im][is]              = qaercw_cur[is][im];
          qaercw_delsub_grow4rnam_tmp[im][is] = qaercw_delsub_grow4rnam[is][im];
        }
      }
      Rename rename;
      rename.mam_rename_1subarea_(
          iscldy_subarea, smallest_dryvol_value, dest_mode_of_mode,   // in
          mean_std_dev, fmode_dist_tail_fac, v2n_lo_rlx, v2n_hi_rlx,  // in
          ln_diameter_tail_fac, num_pairs, diameter_cutoff,           // in
          ln_dia_cutoff,                                              // in
          diameter_threshold, mass_2_vol, dgnum_amode,                // in
          qnum_cur, qaer_cur_tmp, qaer_delsub_grow4rnam_tmp,          // out
          qnumcw_cur, qaercw_cur_tmp, qaercw_delsub_grow4rnam_tmp);   // out

      // copy the output back to the variables
      for(int is = 0; is < nspecies; ++is) {
        for(int im = 0; im < nmodes; ++im) {
          qaer_cur[is][im]                = qaer_cur_tmp[im][is];
          qaer_delsub_grow4rnam[is][im]   = qaer_delsub_grow4rnam_tmp[im][is];
          qaercw_cur[is][im]              = qaercw_cur_tmp[im][is];
          qaercw_delsub_grow4rnam[is][im] = qaercw_delsub_grow4rnam_tmp[im][is];
        }
      }

      //------------------------
      // Accumulate increments
      //------------------------
      for(int im = 0; im < nmodes; ++im) {
        for(int iq = 0; iq < iqtend_rnam(); ++iq) {
          qnum_delaa[im][iq] =
              qnum_delaa[im][iq] + (qnum_cur[im] - qnum_sv1[im]);
        }
      }

      for(int is = 0; is < nspecies; ++is) {
        for(int im = 0; im < nmodes; ++im) {
          for(int iq = 0; iq < iqtend_rnam(); ++iq) {
            qaer_delaa[is][im][iq] =
                qaer_delaa[is][im][iq] + (qaer_cur[is][im] - qaer_sv1[is][im]);
          }
        }
      }

      if(iscldy_subarea) {
        for(int im = 0; im < nmodes; ++im) {
          for(int iq = 0; iq < iqqcwtend_rnam(); ++iq) {
            qnumcw_delaa[im][iq] =
                qnumcw_delaa[im][iq] + (qnumcw_cur[im] - qnumcw_sv1[im]);
          }
        }
        for(int is = 0; is < nspecies; ++is) {
          for(int im = 0; im < nmodes; ++im) {
            for(int iq = 0; iq < iqqcwtend_rnam(); ++iq) {
              qaercw_delaa[is][im][iq] =
                  qaercw_delaa[is][im][iq] +
                  (qaercw_cur[is][im] - qaercw_sv1[is][im]);
            }
          }
        }
      }
    }  // do_rename_sub

    //====================================
    // New particle formation (nucleation)
    //====================================
    if(do_newnuc_sub) {
      copy_1d_array(max_gas(), qgas_cur,  // in
                    qgas_sv1);            // out
      copy_1d_array(nmodes, qnum_cur,     // in
                    qnum_sv1);            // out

      copy_2d_array(nspecies, nmodes, qaer_cur,  // in
                    qaer_sv1);                   // out
#if 0
         call mam_newnuc_1subarea(                                    
            nstep,             lchnk,                                 
            ii,                kk,               jsubarea,            
            latndx,            lonndx,           lund,                
            dtsubstep,                                                
            temp,              pmid,             aircon,              
            zmid,              pblh,             relhum,              
            uptkrate_h2so4,   del_h2so4_gasprod, del_h2so4_aeruptk,   
            n_mode,                                                   
            qgas_cur,          qgas_avg,                              
            qnum_cur,                                                 
            qaer_cur,                                                 
            qwtr_cur,                                                 
            dnclusterdt_substep                                        )

         qgas_delaa(:,iqtend_nnuc) = qgas_delaa(:,iqtend_nnuc) + (qgas_cur - qgas_sv1)
         qnum_delaa(:,iqtend_nnuc) = qnum_delaa(:,iqtend_nnuc) + (qnum_cur - qnum_sv1)
         qaer_delaa(:,:,iqtend_nnuc) = qaer_delaa(:,:,iqtend_nnuc) + (qaer_cur - qaer_sv1)

         misc_vars_aa_sub%ncluster_tend_nnuc_1grid =
         misc_vars_aa_sub%ncluster_tend_nnuc_1grid + dnclusterdt_substep*(dtsubstep/deltat)
#endif
    }  // do_newnuc_sub

    //====================================
    // Coagulation
    //====================================
    if(do_coag_sub) {
      copy_1d_array(nmodes, qnum_cur,  // in
                    qnum_sv1);         // out

      copy_2d_array(nspecies, nmodes, qaer_cur,  // in
                    qaer_sv1);                   // out

      mam4::coagulation::mam_coag_1subarea(
          dtsubstep,                                 // in
          temp, pmid, aircon,                        // in
          dgn_a, dgn_awet, wetdens,                  // in
          qnum_cur, qaer_cur, qaer_delsub_coag_in);  // inout, inout, out

      for(int im = 0; im < nmodes; ++im) {
        qnum_delsub_coag[im] = qnum_cur[im] - qnum_sv1[im];
      }

      for(int is = 0; is < nspecies; ++is) {
        for(int im = 0; im < nmodes; ++im) {
          qaer_delsub_coag[is][im] = qaer_cur[is][im] - qaer_sv1[is][im];
        }
      }

      for(int im = 0; im < nmodes; ++im) {
        for(int iq = 0; iq < iqtend_coag(); ++iq) {
          qnum_delaa[im][iq] = qnum_delaa[im][iq] + qnum_delsub_coag[im];
        }
      }
      for(int is = 0; is < nspecies; ++is) {
        for(int im = 0; im < nmodes; ++im) {
          for(int iq = 0; iq < iqtend_coag(); ++iq) {
            qaer_delaa[is][im][iq] =
                qaer_delaa[is][im][iq] + qaer_delsub_coag[is][im];
          }
        }
      }

    } else {
      assign_2d_array(nspecies, max_agepair, 0.0,  // in
                      qaer_delsub_coag_in);        // out
      assign_2d_array(nspecies, nmodes, 0.0,       // in
                      qaer_delsub_coag);           // out
      assign_1d_array(nmodes, 0.0,                 // in
                      qnum_delsub_coag);           // out

    }  // do_coag_sub

    //====================================
    // primary carbon aging
    //====================================
    const bool do_aging_in_subarea =
        (n_agepair() > 0) &&
        ((!iscldy_subarea) || (iscldy_subarea && do_cond_sub));

    if(do_aging_in_subarea) {
      mam4::aging::mam_pcarbon_aging_1subarea(
          dgn_a,                                         // input
          qnum_cur, qnum_delsub_cond, qnum_delsub_coag,  // in-outs
          qaer_cur, qaer_delsub_cond, qaer_delsub_coag,  // in-outs
          qaer_delsub_coag_in);                          // in-outs
    }                                                    // do_aging_in_subarea

    // The following block has to be placed here (after both condensation and
    // aging) as both can change the values of qnum_delsub_cond and
    // qaer_delsub_cond.

    if(do_cond_sub) {
      for(int im = 0; im < nmodes; ++im) {
        for(int iq = 0; iq < iqtend_cond(); ++iq) {
          qnum_delaa[im][iq] = qnum_delaa[im][iq] + qnum_delsub_cond[im];
        }
      }
      for(int is = 0; is < nspecies; ++is) {
        for(int im = 0; im < nmodes; ++im) {
          for(int iq = 0; iq < iqtend_cond(); ++iq) {
            qaer_delaa[is][im][iq] =
                qaer_delaa[is][im][iq] + qaer_delsub_cond[is][im];
          }
        }
      }
    }  // do_cond_sub
  }    // jtsubstep_loop
       //***********************************

}  // mam_amicphys_1subarea
//--------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void mam_amicphys_1subarea_cloudy(
    const AmicPhysConfig &config, const int nstep, const Real deltat,
    const int jsub, const int nsubarea, const bool iscldy_subarea,
    const Real afracsub, const Real temp, const Real pmid, const Real pdel,
    const Real zmid, const Real pblh, const Real relhum,
    Real dgn_a[AeroConfig::num_modes()], Real dgn_awet[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()],
    const Real qgas1[AeroConfig::num_gas_ids()],
    const Real qgas3[AeroConfig::num_gas_ids()],
    Real qgas4[AeroConfig::num_gas_ids()],
    Real qgas_delaa[AeroConfig::num_gas_ids()][nqtendaa()],
    const Real qnum3[AeroConfig::num_modes()],
    Real qnum4[AeroConfig::num_modes()],
    Real qnum_delaa[AeroConfig::num_modes()][nqtendaa()],
    const Real qaer2[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    const Real qaer3[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer4[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_delaa[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
                   [nqtendaa()],
    const Real qwtr3[AeroConfig::num_modes()],
    Real qwtr4[AeroConfig::num_modes()],
    const Real qnumcw3[AeroConfig::num_modes()],
    Real qnumcw4[AeroConfig::num_modes()],
    Real qnumcw_delaa[AeroConfig::num_modes()][nqqcwtendaa()],
    const Real qaercw2[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    const Real qaercw3[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    Real qaercw4[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    Real qaercw_delaa[AeroConfig::num_gas_ids()][AeroConfig::num_modes()]
                     [nqqcwtendaa()]) {
  //
  // calculates changes to gas and aerosol sub-area TMRs (tracer mixing
  // ratios) qgas3, qaer3, qaercw3, qnum3, qnumcw3 are the current
  // incoming TMRs qgas4, qaer4, qaercw4, qnum4, qnumcw4 are the updated
  // outgoing TMRs
  //
  // when config.do_cond = false, this routine only calculates changes
  // involving
  //    growth from smaller to larger modes (renaming) following cloud
  //    chemistry so gas TMRs are not changed
  // when config.do_cond = true, this routine also calculates changes
  // involving
  //    gas-aerosol exchange (condensation/evaporation)
  //    transfer of particles from hydrophobic modes to hydrophilic modes
  //    (aging)
  //       due to condensation
  // currently this routine does not do
  //    new particle nucleation - because h2so4 gas conc. should be very
  //    low in cloudy air coagulation - because cloud-borne aerosol would
  //    need to be included
  //

  // qXXXN (X=gas,aer,wat,num; N=1:4) are sub-area mixing ratios
  //    XXX=gas - gas species
  //    XXX=aer - aerosol mass  species (excluding water)
  //    XXX=wat - aerosol water
  //    XXX=num - aerosol number
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - current incoming values (before gas-aerosol exchange,
  //    newnuc, coag) N=4 - updated outgoing values (after  gas-aerosol
  //    exchange, newnuc, coag)
  //
  // qXXX_delaa are TMR changes (not tendencies)
  //    for different processes, which are used to produce history output
  // for a clear sub-area, the processes are condensation/evaporation (and
  // associated aging),
  //    renaming, coagulation, and nucleation

  // qxxx_del_yyyy    are mix-ratio changes over full time step (deltat)
  // qxxx_delsub_yyyy are mix-ratio changes over time sub-step (dtsubstep)

  static constexpr int num_gas_ids     = AeroConfig::num_gas_ids();
  static constexpr int num_modes       = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();

  static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  // Turn off nh3 for now.  This is a future enhancement.
  static constexpr int igas_nh3 = -999888777;  // Same as mam_refactor
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_pom = static_cast<int>(AeroId::POM);

  const AeroId gas_to_aer[num_gas_ids] = {AeroId::SOA, AeroId::SO4,
                                          AeroId::None};
  const bool l_gas_condense_to_mode[num_gas_ids][num_modes] = {
      {true, true, true, true},
      {true, true, true, true},
      {false, false, false, false}};
  enum { NA, ANAL, IMPL };
  const int eqn_and_numerics_category[num_gas_ids] = {IMPL, ANAL, ANAL};
  // air molar density (kmol/m3)
  // In order to try to match the results in mam_refactor
  // set r_universal as  [mJ/(mol)] as in mam_refactor.
  // const Real r_universal = Constants::r_gas; // [mJ/(K mol)]
  const Real r_universal = 8.314467591;  // [mJ/(mol)] as in mam_refactor
  const Real aircon      = pmid / (1000 * r_universal * temp);
  const Real alnsg_aer[num_modes] = {0.58778666490211906, 0.47000362924573563,
                                     0.58778666490211906, 0.47000362924573563};
  const Real uptk_rate_factor[num_gas_ids] = {0.81, 1.0, 1.0};

  Real qgas_cur[num_gas_ids];
  for(int i = 0; i < num_gas_ids; ++i) qgas_cur[i] = qgas3[i];
  Real qaer_cur[num_aerosol_ids][num_modes];
  for(int i = 0; i < num_aerosol_ids; ++i)
    for(int j = 0; j < num_modes; ++j) qaer_cur[i][j] = qaer3[i][j];

  Real qnum_cur[num_modes];
  for(int j = 0; j < num_modes; ++j) qnum_cur[j] = qnum3[j];
  Real qwtr_cur[num_modes];
  for(int j = 0; j < num_modes; ++j) qwtr_cur[j] = qwtr3[j];

  Real qnumcw_cur[num_modes];
  for(int j = 0; j < num_modes; ++j) qnumcw_cur[j] = qnumcw3[j];

  Real qaercw_cur[num_gas_ids][num_modes];
  for(int i = 0; i < num_gas_ids; ++i)
    for(int j = 0; j < num_modes; ++j) qaercw_cur[i][j] = qaercw3[i][j];

  Real qgas_netprod_otrproc[num_gas_ids] = {};
  if(config.do_cond && config.gaexch_h2so4_uptake_optaa == 2) {
    for(int igas = 0; igas < num_gas_ids; ++igas) {
      if(igas == igas_h2so4 || igas == igas_nh3) {
        // if gaexch_h2so4_uptake_optaa == 2, then
        //    if qgas increases from pre-gaschem to post-cldchem,
        //       start from the pre-gaschem mix-ratio and add in the
        //       production during the integration
        //    if it decreases,
        //       start from post-cldchem mix-ratio
        // *** currently just do this for h2so4 and nh3
        qgas_netprod_otrproc[igas] = (qgas3[igas] - qgas1[igas]) / deltat;
        if(qgas_netprod_otrproc[igas] >= 0.0)
          qgas_cur[igas] = qgas1[igas];
        else
          qgas_netprod_otrproc[igas] = 0.0;
      }
    }
  }
  Real qgas_del_cond[num_gas_ids]                                      = {};
  Real qgas_del_nnuc[num_gas_ids]                                      = {};
  Real qgas_del_cond_only[num_gas_ids]                                 = {};
  Real qaer_del_cond[num_aerosol_ids][num_modes]                       = {};
  Real qaer_del_rnam[num_aerosol_ids][num_modes]                       = {};
  Real qaer_del_nnuc[num_aerosol_ids][num_modes]                       = {};
  Real qaer_del_coag[num_aerosol_ids][num_modes]                       = {};
  Real qaer_delsub_cond[num_aerosol_ids][num_modes]                    = {};
  Real qaer_delsub_coag[num_aerosol_ids][num_modes]                    = {};
  Real qaer_del_cond_only[num_aerosol_ids][num_modes]                  = {};
  Real qaercw_del_rnam[num_aerosol_ids][num_modes]                     = {};
  Real qnum_del_cond[num_modes]                                        = {};
  Real qnum_del_rnam[num_modes]                                        = {};
  Real qnum_del_nnuc[num_modes]                                        = {};
  Real qnum_del_coag[num_modes]                                        = {};
  Real qnum_delsub_cond[num_modes]                                     = {};
  Real qnum_delsub_coag[num_modes]                                     = {};
  Real qnum_del_cond_only[num_modes]                                   = {};
  Real qnumcw_del_rnam[num_modes]                                      = {};
  Real qaer_delsub_coag_in[num_aerosol_ids][AeroConfig::max_agepair()] = {};

  const int ntsubstep = 1;
  Real dtsubstep      = deltat;
  if(ntsubstep > 1) dtsubstep = deltat / ntsubstep;

  // loop over multiple time sub-steps

  for(int jtsubstep = 1; jtsubstep <= ntsubstep; ++jtsubstep) {
    // gas-aerosol exchange
    Real uptkrate_h2so4                                    = 0.0;
    Real qgas_avg[num_gas_ids]                             = {};
    Real qgas_sv1[num_gas_ids]                             = {};
    Real qnum_sv1[num_modes]                               = {};
    Real qaer_sv1[num_aerosol_ids][num_modes]              = {};
    Real qaer_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};

    if(config.do_cond) {
      const bool l_calc_gas_uptake_coeff   = jtsubstep == 1;
      Real uptkaer[num_gas_ids][num_modes] = {};

      for(int i = 0; i < num_gas_ids; ++i) qgas_sv1[i] = qgas_cur[i];
      for(int i = 0; i < num_modes; ++i) qnum_sv1[i] = qnum_cur[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) qaer_sv1[j][i] = qaer_cur[j][i];

      const int nghq         = 2;
      const int ntot_soamode = 4;
      int niter_out          = 0;
      Real g0_soa_out        = 0;
      // time sub-step
      const Real dtsub_soa_fixed = -1.0;
      gasaerexch::mam_gasaerexch_1subarea(
          nghq, igas_h2so4, igas_nh3, ntot_soamode, gas_to_aer, iaer_so4,
          iaer_pom, l_calc_gas_uptake_coeff, l_gas_condense_to_mode,
          eqn_and_numerics_category, dtsubstep, dtsub_soa_fixed, temp, pmid,
          aircon, num_gas_ids, qgas_cur, qgas_avg, qgas_netprod_otrproc,
          qaer_cur, qnum_cur, dgn_awet, alnsg_aer, uptk_rate_factor, uptkaer,
          uptkrate_h2so4, niter_out, g0_soa_out);

      if(config.newnuc_h2so4_conc_optaa == 11)
        qgas_avg[igas_h2so4] =
            0.5 * (qgas_sv1[igas_h2so4] + qgas_cur[igas_h2so4]);
      else if(config.newnuc_h2so4_conc_optaa == 12)
        qgas_avg[igas_h2so4] = qgas_cur[igas_h2so4];

      for(int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond[i] +=
            (qgas_cur[i] - (qgas_sv1[i] + qgas_netprod_otrproc[i] * dtsubstep));

      for(int i = 0; i < num_modes; ++i)
        qnum_delsub_cond[i] = qnum_cur[i] - qnum_sv1[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_delsub_cond[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];

      // qaer_del_grow4rnam = change in qaer_del_cond during latest
      // condensation calculations
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_delsub_grow4rnam[i][j] = qaer_cur[i][j] - qaer_sv1[i][j];
      for(int i = 0; i < num_gas_ids; ++i)
        qgas_del_cond_only[i] = qgas_del_cond[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_del_cond_only[i][j] = qaer_delsub_cond[i][j];
      for(int i = 0; i < num_modes; ++i)
        qnum_del_cond_only[i] = qnum_delsub_cond[i];

    } else {
      for(int i = 0; i < num_gas_ids; ++i) qgas_avg[i] = qgas_cur[i];
    }
    // renaming after "continuous growth"
    if(config.do_rename) {
      constexpr int nmodes                = AeroConfig::num_modes();
      constexpr int naerosol_species      = AeroConfig::num_aerosol_ids();
      const Real smallest_dryvol_value    = 1.0e-25;  // BAD_CONSTANT
      const int dest_mode_of_mode[nmodes] = {-1, 0, -1, -1};

      Real qnumcw_cur[num_modes]                               = {};
      Real qaercw_cur[num_aerosol_ids][num_modes]              = {};
      Real qaercw_delsub_grow4rnam[num_aerosol_ids][num_modes] = {};
      Real mean_std_dev[nmodes];
      Real fmode_dist_tail_fac[nmodes];
      Real v2n_lo_rlx[nmodes];
      Real v2n_hi_rlx[nmodes];
      Real ln_diameter_tail_fac[nmodes];
      int num_pairs = 0;
      Real diameter_cutoff[nmodes];
      Real ln_dia_cutoff[nmodes];
      Real diameter_threshold[nmodes];
      Real mass_2_vol[naerosol_species] = {0.15,
                                           6.4971751412429377e-002,
                                           0.15,
                                           7.0588235294117650e-003,
                                           3.0789473684210526e-002,
                                           5.1923076923076926e-002,
                                           156.20986883198000};

      rename::find_renaming_pairs(dest_mode_of_mode,     // in
                                  mean_std_dev,          // out
                                  fmode_dist_tail_fac,   // out
                                  v2n_lo_rlx,            // out
                                  v2n_hi_rlx,            // out
                                  ln_diameter_tail_fac,  // out
                                  num_pairs,             // out
                                  diameter_cutoff,       // out
                                  ln_dia_cutoff, diameter_threshold);

      for(int i = 0; i < num_modes; ++i) qnum_sv1[i] = qnum_cur[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i) qaer_sv1[j][i] = qaer_cur[j][i];
      Real dgnum_amode[nmodes];
      for(int m = 0; m < nmodes; ++m) {
        dgnum_amode[m] = modes(m).nom_diameter;
      }

      // qaercw_delsub_grow4rnam = change in qaercw from cloud chemistry
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaercw_delsub_grow4rnam[i][j] =
              (qaercw3[i][j] - qaercw2[i][j]) / ntsubstep;
      Real qnumcw_sv1[num_modes];
      for(int i = 0; i < num_modes; ++i) qnumcw_sv1[i] = qnumcw_cur[i];
      Real qaercw_sv1[num_aerosol_ids][num_modes];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j) qaercw_sv1[i][j] = qaercw_cur[i][j];

      {
        Real qmol_i_cur[num_modes][num_aerosol_ids];
        Real qmol_i_del[num_modes][num_aerosol_ids];
        Real qmol_c_cur[num_modes][num_aerosol_ids];
        Real qmol_c_del[num_modes][num_aerosol_ids];
        for(int j = 0; j < num_aerosol_ids; ++j)
          for(int i = 0; i < num_modes; ++i) {
            qmol_i_cur[i][j] = qaer_cur[j][i];
            qmol_i_del[i][j] = qaer_delsub_grow4rnam[j][i];
            qmol_c_cur[i][j] = qaercw_cur[j][i];
            qmol_c_del[i][j] = qaercw_delsub_grow4rnam[j][i];
          }

        Rename rename;
        rename.mam_rename_1subarea_(
            iscldy_subarea, smallest_dryvol_value, dest_mode_of_mode,
            mean_std_dev, fmode_dist_tail_fac, v2n_lo_rlx, v2n_hi_rlx,
            ln_diameter_tail_fac, num_pairs, diameter_cutoff, ln_dia_cutoff,
            diameter_threshold, mass_2_vol, dgnum_amode, qnum_cur, qmol_i_cur,
            qmol_i_del, qnumcw_cur, qmol_c_cur, qmol_c_del);

        for(int j = 0; j < num_aerosol_ids; ++j)
          for(int i = 0; i < num_modes; ++i) {
            qaer_cur[j][i]                = qmol_i_cur[i][j];
            qaer_delsub_grow4rnam[j][i]   = qmol_i_del[i][j];
            qaercw_cur[j][i]              = qmol_c_cur[i][j];
            qaercw_delsub_grow4rnam[j][i] = qmol_c_del[i][j];
          }
      }
      for(int i = 0; i < num_modes; ++i)
        qnum_del_rnam[i] += qnum_cur[i] - qnum_sv1[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaer_del_rnam[i][j] += qaer_cur[i][j] - qaer_sv1[i][j];
      for(int i = 0; i < num_modes; ++i)
        qnumcw_del_rnam[i] += qnumcw_cur[i] - qnumcw_sv1[i];
      for(int i = 0; i < num_aerosol_ids; ++i)
        for(int j = 0; j < num_modes; ++j)
          qaercw_del_rnam[i][j] += qaercw_cur[i][j] - qaercw_sv1[i][j];
    }

    // primary carbon aging
    if(config.do_cond) {
      aging::mam_pcarbon_aging_1subarea(
          dgn_a, qnum_cur, qnum_delsub_cond, qnum_delsub_coag, qaer_cur,
          qaer_delsub_cond, qaer_delsub_coag, qaer_delsub_coag_in);
    }
    // accumulate sub-step q-dels
    if(config.do_cond) {
      for(int i = 0; i < num_modes; ++i)
        qnum_del_cond[i] += qnum_delsub_cond[i];
      for(int j = 0; j < num_aerosol_ids; ++j)
        for(int i = 0; i < num_modes; ++i)
          qaer_del_cond[j][i] += qaer_delsub_cond[j][i];
    }
  }
  // final mix ratios
  for(int i = 0; i < num_gas_ids; ++i) qgas4[i] = qgas_cur[i];
  for(int j = 0; j < num_aerosol_ids; ++j)
    for(int i = 0; i < num_modes; ++i) qaer4[j][i] = qaer_cur[j][i];
  for(int i = 0; i < num_modes; ++i) qnum4[i] = qnum_cur[i];
  for(int i = 0; i < num_modes; ++i) qwtr4[i] = qwtr_cur[i];
  for(int i = 0; i < num_modes; ++i) qnumcw4[i] = qnumcw_cur[i];
  for(int i = 0; i < num_gas_ids; ++i)
    for(int j = 0; j < num_modes; ++j) qaercw4[i][j] = qaercw_cur[i][j];

  // final mix ratio changes
  for(int i = 0; i < num_gas_ids; ++i) {
    qgas_delaa[i][iqtend_cond()]      = qgas_del_cond[i];
    qgas_delaa[i][iqtend_rnam()]      = 0.0;
    qgas_delaa[i][iqtend_nnuc()]      = qgas_del_nnuc[i];
    qgas_delaa[i][iqtend_coag()]      = 0.0;
    qgas_delaa[i][iqtend_cond_only()] = qgas_del_cond_only[i];
  }
  for(int i = 0; i < num_modes; ++i) {
    qnum_delaa[i][iqtend_cond()]      = qnum_del_cond[i];
    qnum_delaa[i][iqtend_rnam()]      = qnum_del_rnam[i];
    qnum_delaa[i][iqtend_nnuc()]      = qnum_del_nnuc[i];
    qnum_delaa[i][iqtend_coag()]      = qnum_del_coag[i];
    qnum_delaa[i][iqtend_cond_only()] = qnum_del_cond_only[i];
  }
  for(int j = 0; j < num_aerosol_ids; ++j) {
    for(int i = 0; i < num_modes; ++i) {
      qaer_delaa[j][i][iqtend_cond()]      = qaer_del_cond[j][i];
      qaer_delaa[j][i][iqtend_rnam()]      = qaer_del_rnam[j][i];
      qaer_delaa[j][i][iqtend_nnuc()]      = qaer_del_nnuc[j][i];
      qaer_delaa[j][i][iqtend_coag()]      = qaer_del_coag[j][i];
      qaer_delaa[j][i][iqtend_cond_only()] = qaer_del_cond_only[j][i];
    }
  }
  for(int i = 0; i < num_modes; ++i)
    qnumcw_delaa[i][iqqcwtend_rnam()] = qnumcw_del_rnam[i];
  for(int i = 0; i < num_aerosol_ids; ++i)
    for(int j = 0; j < num_modes; ++j)
      qaercw_delaa[i][j][iqqcwtend_rnam()] = qaercw_del_rnam[i][j];
}

KOKKOS_INLINE_FUNCTION
void mam_amicphys_1gridcell(
    const int ii, const int kk, const AmicPhysConfig &config, const int nstep,
    const Real deltat, const int nsubarea, const int ncldy_subarea,
    const bool (&iscldy_subarea)[maxsubarea()],
    const Real (&afracsub)[maxsubarea()], const Real temp, const Real pmid,
    const Real pdel, const Real zmid, const Real pblh,
    const Real (&relhumsub)[maxsubarea()],
    Real (&dgn_a)[AeroConfig::num_modes()],
    Real (&dgn_awet)[AeroConfig::num_modes()],
    Real (&wetdens)[AeroConfig::num_modes()],
    const Real (&qsub1)[gas_pcnst()][maxsubarea()],
    const Real (&qsub2)[gas_pcnst()][maxsubarea()],
    const Real (&qqcwsub2)[gas_pcnst()][maxsubarea()],
    const Real (&qsub3)[gas_pcnst()][maxsubarea()],
    const Real (&qqcwsub3)[gas_pcnst()][maxsubarea()],
    Real (&qaerwatsub3)[AeroConfig::num_modes()][maxsubarea()],
    Real (&qsub4)[gas_pcnst()][maxsubarea()],
    Real (&qqcwsub4)[gas_pcnst()][maxsubarea()],
    Real (&qaerwatsub4)[AeroConfig::num_modes()][maxsubarea()],
    Real (&qsub_tendaa)[gas_pcnst()][nqtendaa()][maxsubarea()],
    Real (&qqcwsub_tendaa)[gas_pcnst()][nqqcwtendaa()][maxsubarea()]) {
  //
  // calculates changes to gas and aerosol sub-area TMRs (tracer mixing
  // ratios) qsub3 and qqcwsub3 are the incoming current TMRs qsub4 and
  // qqcwsub4 are the outgoing updated TMRs
  //
  // qsubN and qqcwsubN (N=1:4) are tracer mixing ratios (TMRs, mol/mol or
  // #/kmol) in sub-areas
  //    currently there are just clear and cloudy sub-areas
  //    the N=1:4 have same meanings as for qgcmN
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - incoming values (before gas-aerosol exchange, newnuc, coag)
  //    N=4 - outgoing values (after  gas-aerosol exchange, newnuc, coag)
  // qsub_tendaa and qqcwsub_tendaa are TMR tendencies
  //    for different processes, which are used to produce history output
  // the processes are condensation/evaporation (and associated aging),
  //    renaming, coagulation, and nucleation

  constexpr int mdo_gaexch_cldy_subarea = 0;
  static constexpr int num_gas_ids      = AeroConfig::num_gas_ids();
  static constexpr int num_modes        = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids  = AeroConfig::num_aerosol_ids();

  // the q--4 values will be equal to q--3 values unless they get changed
  for(int i = 0; i < num_gas_ids; ++i)
    for(int j = 1; j <= maxsubarea(); ++j) {
      qsub4[i][j]    = qsub3[i][j];
      qqcwsub4[i][j] = qqcwsub3[i][j];
    }
  for(int i = 0; i < num_modes; ++i)
    for(int j = 1; j <= maxsubarea(); ++j)
      qaerwatsub4[i][j] = qaerwatsub3[i][j];
  for(int i = 0; i < num_gas_ids; ++i)
    for(int j = 0; j < nqtendaa(); ++j)
      for(int kk = 1; kk <= maxsubarea(); ++kk) qsub_tendaa[i][j][kk] = 0;
  for(int i = 0; i < num_gas_ids; ++i)
    for(int j = 0; j < nqqcwtendaa(); ++j)
      for(int kk = 1; kk <= maxsubarea(); ++kk) qqcwsub_tendaa[i][j][kk] = 0;

  for(int jsub = 1; jsub <= nsubarea; ++jsub) {
    AmicPhysConfig sub_config = config;
    // FIXME: Remove the code below after removing
    // mam_amicphys_1subarea_cloudy and mam_amicphys_1subarea_clear
    if(iscldy_subarea[jsub]) {
      sub_config.do_cond   = config.do_cond;
      sub_config.do_rename = config.do_rename;
      sub_config.do_newnuc = false;
      sub_config.do_coag   = false;
      if(mdo_gaexch_cldy_subarea <= 0) sub_config.do_cond = false;
    } else {
      sub_config.do_cond   = config.do_cond;
      sub_config.do_rename = config.do_rename;
      sub_config.do_newnuc = config.do_newnuc;
      sub_config.do_coag   = config.do_coag;
    }
    // FIXME: remove the code above ^^^
    bool do_cond;
    bool do_rename;
    bool do_newnuc;
    bool do_coag;
    if(iscldy_subarea[jsub]) {
      do_cond   = config.do_cond;
      do_rename = config.do_rename;
      do_newnuc = false;
      do_coag   = false;
      if(mdo_gaexch_cldy_subarea <= 0) do_cond = false;
    } else {
      do_cond   = config.do_cond;
      do_rename = config.do_rename;
      do_newnuc = config.do_newnuc;
      do_coag   = config.do_coag;
    }

    if(kk == 48)
      printf("iscldy_subarea:%s, %i\n", iscldy_subarea[jsub] ? "true" : "false",
             jsub);
    const bool do_map_gas_sub = do_cond || do_newnuc;

    // map incoming sub-area mix-ratios to gas/aer/num arrays
    // Initialized to zero; C++ automatically initialze to
    // all elements to zero if fisrt is initialized to zero as
    // below
    Real qgas1[max_gas()] = {0};
    Real qgas2[max_gas()] = {0};
    Real qgas3[max_gas()] = {0};
    Real qgas4[max_gas()] = {0};
    for(int igas = 0; igas < max_gas(); ++igas)
      if(do_map_gas_sub) {
        // for cldy subarea, only do gases if doing gaexch
        for(int igas = 0; igas < max_gas(); ++igas) {
          const int l = lmap_gas(igas);
          qgas1[igas] = qsub1[l][jsub] * fcvt_gas(igas);
          qgas2[igas] = qsub2[l][jsub] * fcvt_gas(igas);
          qgas3[igas] = qsub3[l][jsub] * fcvt_gas(igas);
          qgas4[igas] = qgas3[igas];
          if(kk == 48)
            printf("qgas4:%0.15e,%0.15e,%0.15e,%0.15e,%i,%i, %i\n", qgas4[igas],
                   qgas3[igas], qsub3[l][jsub], fcvt_gas(igas), l, igas, jsub);
        }
      }

    Real qaer2[num_aerosol_ids][num_modes] = {0};
    Real qnum2[num_modes]                  = {0};
    Real qaer3[num_aerosol_ids][num_modes] = {0};
    Real qnum3[num_modes]                  = {0};
    Real qaer4[num_aerosol_ids][num_modes] = {0};
    Real qnum4[num_modes]                  = {0};
    Real qwtr3[num_modes]                  = {0};
    Real qwtr4[num_modes]                  = {0};

    for(int imode = 0; imode < num_modes; ++imode) {
      const int ln = lmap_num(imode);
      qnum2[imode] = qsub2[ln][jsub] * fcvt_num();
      qnum3[imode] = qsub3[ln][jsub] * fcvt_num();
      qnum4[imode] = qnum3[imode];
      for(int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
        const int la = lmap_aer(iaer, imode);
        if(la > 0) {
          qaer2[iaer][imode] = qsub2[la][jsub] * fcvt_aer(iaer);
          qaer3[iaer][imode] = qsub3[la][jsub] * fcvt_aer(iaer);
          qaer4[iaer][imode] = qaer3[iaer][imode];
        }
      }  // for iaer
      qwtr3[imode] = qaerwatsub3[imode][jsub] * fcvt_wtr();
      qwtr4[imode] = qwtr3[imode];
    }  // for imode
    if(kk == 48) {
      for(int imode = 0; imode < num_modes; ++imode) {
        printf("NUM_1:%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%i\n",
               qnum2[imode], qnum3[imode], qnum4[imode], qwtr3[imode],
               qwtr4[imode], fcvt_wtr(), fcvt_num(), imode);
        int endind = num_aerosol_ids;
        if(imode == 1) endind = 4;
        if(imode == 3) endind = 3;
        for(int icnst = 0; icnst < endind; ++icnst) {
          printf("qaer_1:%.15e,%.15e,%.15e,%.15e,%i, %i\n", qaer2[icnst][imode],
                 qaer3[icnst][imode], qaer4[icnst][imode], fcvt_aer(icnst),
                 icnst, imode);
        }
      }
    }

    Real qaercw2[num_aerosol_ids][num_modes] = {0};
    Real qnumcw2[num_modes]                  = {0};
    Real qaercw3[num_aerosol_ids][num_modes] = {0};
    Real qnumcw3[num_modes]                  = {0};
    Real qaercw4[num_aerosol_ids][num_modes] = {0};
    Real qnumcw4[num_modes]                  = {0};

    // FIXME: Remove the following assignment code:
    for(int imode = 0; imode < num_modes; ++imode) {
      qnumcw2[imode] = 999;
      qnumcw3[imode] = 999;
      qnumcw4[imode] = 999;
      for(int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
        qaercw2[iaer][imode] = 888;
        qaercw3[iaer][imode] = 888;
        qaercw4[iaer][imode] = 888;
      }  // iaer
    }    // imode

    // FIXME: Remove code above till FIXME
    if(iscldy_subarea[jsub]) {
      // only do cloud-borne for cloudy
      for(int imode = 0; imode < num_modes; ++imode) {
        int ln         = lmap_num(imode);
        qnumcw2[imode] = qqcwsub2[ln][jsub] * fcvt_num();
        qnumcw3[imode] = qqcwsub3[ln][jsub] * fcvt_num();
        qnumcw4[imode] = qnumcw3[imode];
        for(int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
          int la = lmap_aer(iaer, imode);
          if(la > 0) {
            qaercw2[iaer][imode] = qqcwsub2[la][jsub] * fcvt_aer(iaer);
            qaercw3[iaer][imode] = qqcwsub3[la][jsub] * fcvt_aer(iaer);
            if(kk == 48)
              printf("qaercw3:%.15e,%.15e,%.15e,%i, %i\n", qaercw3[iaer][imode],
                     qqcwsub3[la][jsub], fcvt_aer(iaer), iaer, imode);
            qaercw4[iaer][imode] = qaercw3[iaer][imode];
          }  // la
        }    // iaer
      }      // imode
    }        // iscldy_subarea

    if(kk == 48) {
      for(int imode = 0; imode < num_modes; ++imode) {
        printf("NUMCW_1:%.15e,%.15e,%.15e,%i\n", qnumcw2[imode], qnumcw3[imode],
               qnumcw4[imode], imode);
        int endind = num_aerosol_ids;
        if(imode == 1) endind = 4;
        if(imode == 3) endind = 3;
        for(int iaer = 0; iaer < endind; ++iaer) {
          printf("qaerCW_1:%.15e,%.15e,%.15e,%i, %i\n", qaercw2[iaer][imode],
                 qaercw3[iaer][imode], qaercw4[iaer][imode], iaer, imode);
        }
      }
    }

    Real qgas_delaa[max_gas()][nqtendaa()]                       = {};
    Real qnum_delaa[num_modes][nqtendaa()]                       = {};
    Real qnumcw_delaa[num_modes][nqqcwtendaa()]                  = {};
    Real qaer_delaa[num_aerosol_ids][num_modes][nqtendaa()]      = {};
    Real qaercw_delaa[num_aerosol_ids][num_modes][nqqcwtendaa()] = {};

    mam_amicphys_1subarea(
        config.gaexch_h2so4_uptake_optaa, do_cond, do_rename, do_newnuc,
        do_coag, ii, kk, deltat, jsub, nsubarea, iscldy_subarea[jsub],
        afracsub[jsub], temp, pmid, pdel, zmid, pblh, relhumsub[jsub], dgn_a,
        dgn_awet, wetdens, qgas1, qgas3, qgas4, qgas_delaa, qnum3, qnum4,
        qnum_delaa, qaer2, qaer3, qaer4, qaer_delaa, qwtr3, qwtr4, qnumcw3,
        qnumcw4, qnumcw_delaa, qaercw2, qaercw3, qaercw4, qaercw_delaa);

    if(iscldy_subarea[jsub]) {
      mam_amicphys_1subarea_cloudy(
          sub_config, nstep, deltat, jsub, nsubarea, iscldy_subarea[jsub],
          afracsub[jsub], temp, pmid, pdel, zmid, pblh, relhumsub[jsub], dgn_a,
          dgn_awet, wetdens, qgas1, qgas3, qgas4, qgas_delaa, qnum3, qnum4,
          qnum_delaa, qaer2, qaer3, qaer4, qaer_delaa, qwtr3, qwtr4, qnumcw3,
          qnumcw4, qnumcw_delaa, qaercw2, qaercw3, qaercw4, qaercw_delaa);
    } else {
      mam_amicphys_1subarea_clear(
          sub_config, nstep, deltat, jsub, nsubarea, iscldy_subarea[jsub],
          afracsub[jsub], temp, pmid, pdel, zmid, pblh, relhumsub[jsub], dgn_a,
          dgn_awet, wetdens, qgas1, qgas3, qgas4, qgas_delaa, qnum3, qnum4,
          qnum_delaa, qaer3, qaer4, qaer_delaa, qwtr3, qwtr4);
      // map gas/aer/num arrays (mix-ratio and del=change) back to
      // sub-area arrays

      if(do_map_gas_sub) {
        for(int igas = 0; igas < max_gas(); ++igas) {
          const int l    = lmap_gas(igas);
          qsub4[l][jsub] = qgas4[igas] / fcvt_gas(igas);
          if(kk == 48)
            printf("qsub4:%0.15e,%0.15e,%0.15e,%i,%i,%i\n", qsub4[l][jsub],
                   qgas4[igas], fcvt_gas(igas), l, jsub, igas);
          for(int i = 0; i < nqtendaa(); ++i)
            qsub_tendaa[l][i][jsub] =
                qgas_delaa[igas][i] / (fcvt_gas(igas) * deltat);
        }
      }
      for(int n = 0; n < num_modes; ++n) {
        qsub4[n][jsub] = qnum4[n] / fcvt_num();
        for(int i = 0; i < nqtendaa(); ++i)
          qsub_tendaa[n][i][jsub] = qnum_delaa[n][i] / (fcvt_num() * deltat);
        for(int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
          qsub4[iaer][jsub] = qaer4[iaer][n] / fcvt_aer(iaer);
          for(int i = 0; i < nqtendaa(); ++i)
            qsub_tendaa[iaer][i][jsub] =
                qaer_delaa[iaer][n][i] / (fcvt_aer(iaer) * deltat);
        }
        qaerwatsub4[n][jsub] = qwtr4[n] / fcvt_wtr();

        if(iscldy_subarea[jsub]) {
          qqcwsub4[n][jsub] = qnumcw4[n] / fcvt_num();
          for(int i = 0; i < nqqcwtendaa(); ++i)
            qqcwsub_tendaa[n][i][jsub] =
                qnumcw_delaa[n][i] / (fcvt_num() * deltat);
          for(int iaer = 0; iaer < num_aerosol_ids; ++iaer) {
            qqcwsub4[iaer][jsub] = qaercw4[iaer][n] / fcvt_aer(iaer);
            for(int i = 0; i < nqqcwtendaa(); ++i)
              qqcwsub_tendaa[iaer][i][jsub] =
                  qaercw_delaa[iaer][n][i] / (fcvt_aer(iaer) * deltat);
          }
        }
      }
    }
  }
}

}  // anonymous namespace

KOKKOS_INLINE_FUNCTION
void modal_aero_amicphys_intr(int ii, int kk, const AmicPhysConfig &config,
                              const int nstep, const Real deltat,
                              const Real temp, const Real pmid, const Real pdel,
                              const Real zm, const Real pblh, const Real qv,
                              const Real cld, Real q[gas_pcnst()],
                              Real qqcw[gas_pcnst()],
                              const Real q_pregaschem[gas_pcnst()],
                              const Real q_precldchem[gas_pcnst()],
                              const Real qqcw_precldchem[gas_pcnst()],
                              Real q_tendbb[gas_pcnst()][nqtendbb()],
                              Real qqcw_tendbb[gas_pcnst()][nqtendbb()],
                              const Real dgncur_a[AeroConfig::num_modes()],
                              const Real dgncur_awet[AeroConfig::num_modes()],
                              const Real wetdens_host[AeroConfig::num_modes()],
                              Real qaerwat[AeroConfig::num_modes()]) {
  /*
      nstep                ! model time-step number
      nqtendbb             ! dimension for q_tendbb
      nqqcwtendbb          ! dimension f
      deltat               !
      q(ncol,pver,pcnstxx) ! current tracer mixing ratios (TMRs)
                              these values are updated (so out /= in)
                           *** MUST BE  #/kmol-air for number
                           *** MUST BE mol/mol-air for mass
                           *** NOTE ncol dimension
      qqcw(ncol,pver,pcnstxx)
                            like q but for cloud-borner tracers
                            these values are updated
      q_pregaschem(ncol,pver,pcnstxx)    ! q TMRs    before gas-phase
    chemistry q_precldchem(ncol,pver,pcnstxx)    ! q TMRs    before cloud
    chemistry qqcw_precldchem(ncol,pver,pcnstxx) ! qqcw TMRs before cloud
    chemistry q_tendbb(ncol,pver,pcnstxx,nqtendbb())    ! TMR tendencies for
    box-model diagnostic output qqcw_tendbb(ncol,pver,pcnstx t(pcols,pver) !
    temperature at model levels (K) pmid(pcols,pver)     ! pressure at model
    level centers (Pa) pdel(pcols,pver)     ! pressure thickness of levels
    (Pa) zm(pcols,pver)       ! altitude (above ground) at level centers (m)
    pblh(pcols)          ! planetary boundary layer depth (m)
    qv(pcols,pver)       ! specific humidity (kg/kg)
    cld(ncol,pver)       ! cloud fraction (-) *** NOTE ncol dimension
    dgncur_a(pcols,pver,ntot_amode)
    dgncur_awet(pcols,pver,ntot_amode)
                                        ! dry & wet geo. mean dia. (m) of
    number distrib. wetdens_host(pcols,pver,ntot_amode) ! interstitial
    aerosol wet density (kg/m3)

      qaerwat(pcols,pver,ntot_amode    aerosol water mixing ratio (kg/kg,
    NOT mol/mol)

  */

  // !DESCRIPTION:
  // calculates changes to gas and aerosol TMRs (tracer mixing ratios) from
  //    gas-aerosol exchange (condensation/evaporation)
  //    growth from smaller to larger modes (renaming) due to both
  //       condensation and cloud chemistry
  //    new particle nucleation
  //    coagulation
  //    transfer of particles from hydrophobic modes to hydrophilic modes
  //    (aging)
  //       due to condensation and coagulation
  //
  // the incoming mixing ratios (q and qqcw) are updated before output
  //
  // !REVISION HISTORY:
  //   RCE 07.04.13:  Adapted from earlier version of CAM5 modal aerosol
  //   routines
  //                  for these processes
  //

  static constexpr int num_modes = AeroConfig::num_modes();

  // qgcmN and qqcwgcmN (N=1:4) are grid-cell mean tracer mixing ratios
  // (TMRs, mol/mol or #/kmol)
  //    N=1 - before gas-phase chemistry
  //    N=2 - before cloud chemistry
  //    N=3 - incoming values (before gas-aerosol exchange, newnuc, coag)
  //    N=4 - outgoing values (after  gas-aerosol exchange, newnuc, coag)

  // qsubN and qqcwsubN (N=1:4) are TMRs in sub-areas
  //    currently there are just clear and cloudy sub-areas
  //    the N=1:4 have same meanings as for qgcmN

  // q_coltendaa and qqcw_coltendaa are column-integrated tendencies
  //    for different processes, which are output to history
  // the processes are condensation/evaporation (and associated aging),
  //    renaming, coagulation, and nucleation

  for(int i = 0; i < gas_pcnst(); ++i)
    for(int j = 0; j < nqtendbb(); ++j)
      q_tendbb[i][j] = 0.0, qqcw_tendbb[i][j] = 0.0;

  // get saturation mixing ratio
  //     call qsat( t(1:ncol,1:pver), pmid(1:ncol,1:pvnner), &
  //               ev_sat(1:ncol,1:pver), qv_sat(1:ncol,1:pver) )
  const Real epsqs = haero::Constants::weight_ratio_h2o_air;
  // Saturation vapor pressure
  const Real ev_sat = conversions::vapor_saturation_pressure_magnus(temp, pmid);
  // Saturation specific humidity
  Real qv_sat = epsqs * ev_sat /
                (pmid - (1 - epsqs) * ev_sat);  // BALLI: make it a const var
  if(kk == 48) {
    const Real qv_sat_hardwired = 7.614546931278814E-003;
    qv_sat                      = qv_sat_hardwired;
  }

  int nsubarea;       // total # of subareas to do calculations for
  int ncldy_subarea;  // total # of cloudy subareas
  int jclea, jcldy;   // indices of the clear and cloudy subareas
  bool iscldy_subarea[maxsubarea()];  // whether a subarea is cloudy
  Real afracsub[maxsubarea()];        // area fraction of each active
                                      // subarea[unitless]
  Real fcldy;                         // in  cloudy fraction of the grid cell
  Real fclea;                         // in clear  fraction of the grid cell

  setup_subareas(kk, cld,                                  // in
                 nsubarea, ncldy_subarea, jclea, jcldy,    // out
                 iscldy_subarea, afracsub, fclea, fcldy);  // out

  const Real relhumgcm = haero::max(0.0, haero::min(1.0, qv / qv_sat));
  if(kk == 48) {
    printf("QSAT:%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", qv_sat, qv, pmid, temp,
           epsqs, ev_sat);
  }

  Real relhumsub[maxsubarea()];
  set_subarea_rh(ncldy_subarea, jclea, jcldy, afracsub, relhumgcm,  // in
                 relhumsub);                                        // out

  //-------------------------------
  // Set aerosol water in subareas
  //-------------------------------
  // Notes from Dick Easter/Steve Ghan: how to treat aerosol water in
  // subareas needs more work/thinking Currently modal_aero_water_uptake
  // calculates qaerwat using the grid-cell mean interstital-aerosol
  // mix-rats and the clear-area RH. aerosol water mixing ratios (mol/mol)
  Real qaerwatsub3[AeroConfig::num_modes()][maxsubarea()] = {
      0};  // FIXME: can it be constexpr??

  //-------------------------------------------------------------------------
  // Set gases, interstitial aerosols, and cloud-borne aerosols in subareas
  //-------------------------------------------------------------------------
  // Copy grid cell mean mixing ratios; clip negative values if any.
  Real qgcm1[gas_pcnst()], qgcm2[gas_pcnst()], qgcm3[gas_pcnst()];
  Real qqcwgcm2[gas_pcnst()], qqcwgcm3[gas_pcnst()];  // cld borne aerosols
  for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
    // Gases and interstitial aerosols
    qgcm1[icnst] = haero::max(0, q_pregaschem[icnst]);
    qgcm2[icnst] = haero::max(0, q_precldchem[icnst]);
    qgcm3[icnst] = haero::max(0, q[icnst]);

    // Cloud-borne aerosols
    qqcwgcm2[icnst] = haero::max(0, qqcw_precldchem[icnst]);
    qqcwgcm3[icnst] = haero::max(0, qqcw[icnst]);
  }
  if(kk == 48) {
    for(int icnst = 0; icnst < 6; ++icnst) {
      printf("amic2a:%.15e,%.15e,%.15e, %i\n", qgcm1[icnst], qgcm2[icnst],
             qgcm3[icnst], icnst);
    }
  }
  // Partition grid cell mean to subareas
  Real qsub1[gas_pcnst()][maxsubarea()];
  Real qsub2[gas_pcnst()][maxsubarea()];
  Real qsub3[gas_pcnst()][maxsubarea()];
  Real qqcwsub1[gas_pcnst()][maxsubarea()];
  Real qqcwsub2[gas_pcnst()][maxsubarea()];
  Real qqcwsub3[gas_pcnst()][maxsubarea()];

  set_subarea_gases_and_aerosols(kk, nsubarea, jclea, jcldy, fclea,
                                 fcldy,  // in
                                 qgcm1, qgcm2, qqcwgcm2, qgcm3,
                                 qqcwgcm3,                       // in
                                 qsub1, qsub2, qqcwsub2, qsub3,  // out
                                 qqcwsub3);                      // out
  if(kk == 48) {
    printf("amic1:%i,%i,%i,%i,%s,%s,%.15e,%.15e,%.15e,%.15e\n", nsubarea,
           ncldy_subarea, jclea, jcldy, iscldy_subarea[1] ? "true" : "false",
           iscldy_subarea[2] ? "true" : "false", afracsub[1], afracsub[2],
           relhumsub[1], relhumsub[2]);
  }

  if(kk == 48) {
    for(int ii = 0; ii < AeroConfig::num_modes(); ++ii) {
      for(int jj = 0; jj < maxsubarea(); ++jj) {
        // printf("amic2:%.15e,%i, %i\n", qaerwatsub3[ii][jj], ii, jj);
      }
    }
  }

  if(kk == 48) {
    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      // printf("lmapcc_aa:%i, %i\n", lmapcc_all(icnst), icnst);
    }
  }
  if(kk == 48) {
    for(int jsub = 1; jsub < maxsubarea(); ++jsub) {
      for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
        printf("amic3:%.15e,%.15e,%.15e,%.15e,%.15e,%i, %i\n",
               qsub1[icnst][jsub], qsub2[icnst][jsub], qqcwsub2[icnst][jsub],
               qsub3[icnst][jsub], qqcwsub3[icnst][jsub], icnst + 1, jsub);
      }
    }
  }

  //  Initialize the "after-amicphys" values
  Real qsub4[gas_pcnst()][maxsubarea()]                   = {};
  Real qqcwsub4[gas_pcnst()][maxsubarea()]                = {};
  Real qaerwatsub4[AeroConfig::num_modes()][maxsubarea()] = {};

  //
  // start integration
  //
  Real dgn_a[num_modes]    = {0};
  Real dgn_awet[num_modes] = {0};
  Real wetdens[num_modes]  = {1000};

  for(int n = 0; n < num_modes; ++n) {
    dgn_a[n]    = dgncur_a[n];
    dgn_awet[n] = dgncur_awet[n];
    wetdens[n]  = haero::max(1000.0, wetdens_host[n]);
    if(kk == 48) {
      printf("dgn_a:%0.15e, %i\n", dgn_a[n], n);
      printf("dgn_awet:%0.15e, %i\n", dgn_awet[n], n);
      printf("wetdens:%0.15e, %i\n", wetdens[n], n);
    }
  }

  Real qsub_tendaa[gas_pcnst()][nqtendaa()][maxsubarea()]       = {};
  Real qqcwsub_tendaa[gas_pcnst()][nqqcwtendaa()][maxsubarea()] = {};
  mam_amicphys_1gridcell(
      ii, kk, config, nstep, deltat, nsubarea, ncldy_subarea,  // in
      iscldy_subarea, afracsub, temp, pmid, pdel, zm,          // in
      pblh,                                                    // in
      relhumsub,                                               // in
      // FIXME: dgn_a, dgn_awet, wetdens seems like "in", confirm it
      dgn_a, dgn_awet, wetdens,                   // inout
      qsub1, qsub2, qqcwsub2, qsub3, qqcwsub3,    // in
      qaerwatsub3, qsub4, qqcwsub4, qaerwatsub4,  // inout
      qsub_tendaa, qqcwsub_tendaa);               // inout
  if(kk == 48) {
    for(int n = 0; n < num_modes; ++n) {
      printf("AFT:dgn_a:%0.15e, %i\n", dgn_a[n], n);
      printf("AFT:dgn_awet:%0.15e, %i\n", dgn_awet[n], n);
      printf("AFT:wetdens:%0.15e, %i\n", wetdens[n], n);
    }
    for(int jsub = 1; jsub < maxsubarea(); ++jsub) {
      for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
        printf("amic4:%.15e,%.15e,%i, %i\n", qsub4[icnst][jsub],
               qqcwsub4[icnst][jsub], icnst + 1, jsub);
      }
      for(int icnst = 0; icnst < AeroConfig::num_modes(); ++icnst) {
        printf("amic4:%.15e,%.15e,%i, %i\n", qaerwatsub3[icnst][jsub],
               qaerwatsub4[icnst][jsub], icnst + 1, jsub);
      }
    }
  }

  //
  // form new grid-mean mix-ratios
  Real qgcm4[gas_pcnst()];
  Real qgcm_tendaa[gas_pcnst()][nqtendaa()];
  Real qaerwatgcm4[num_modes];
  if(nsubarea == 1) {
    for(int i = 0; i < gas_pcnst(); ++i) {
      qgcm4[i] = qsub4[i][0];
    }
    for(int i = 0; i < gas_pcnst(); ++i)
      for(int j = 0; j < nqtendaa(); ++j)
        qgcm_tendaa[i][j] = qsub_tendaa[i][j][0];
    for(int i = 0; i < num_modes; ++i) qaerwatgcm4[i] = qaerwatsub4[i][0];
  } else {
    for(int i = 0; i < gas_pcnst(); ++i) qgcm4[i] = 0.0;
    for(int i = 0; i < gas_pcnst(); ++i)
      for(int j = 0; j < nqtendaa(); ++j) qgcm_tendaa[i][j] = 0.0;
    for(int n = 0; n <= nsubarea; ++n) {
      for(int i = 0; i < gas_pcnst(); ++i) {
        qgcm4[i] += qsub4[i][n] * afracsub[n];
      }
      for(int i = 0; i < gas_pcnst(); ++i)
        for(int j = 0; j < nqtendaa(); ++j)
          qgcm_tendaa[i][j] =
              qgcm_tendaa[i][j] + qsub_tendaa[i][j][n] * afracsub[n];
    }
    for(int i = 0; i < num_modes; ++i)
      // for aerosol water use the clear sub-area value
      qaerwatgcm4[i] = qaerwatsub4[i][jclea - 1];
  }
  Real qqcwgcm4[gas_pcnst()];
  Real qqcwgcm_tendaa[gas_pcnst()][nqqcwtendaa()];
  if(ncldy_subarea <= 0) {
    for(int i = 0; i < gas_pcnst(); ++i) qqcwgcm4[i] = haero::max(0.0, qqcw[i]);
    for(int i = 0; i < gas_pcnst(); ++i)
      for(int j = 0; j < nqqcwtendaa(); ++j) qqcwgcm_tendaa[i][j] = 0.0;
  } else if(nsubarea == 1) {
    for(int i = 0; i < gas_pcnst(); ++i) qqcwgcm4[i] = qqcwsub4[i][0];
    for(int i = 0; i < gas_pcnst(); ++i)
      for(int j = 0; j < nqqcwtendaa(); ++j)
        qqcwgcm_tendaa[i][j] = qqcwsub_tendaa[i][j][0];
  } else {
    for(int i = 0; i < gas_pcnst(); ++i) qqcwgcm4[i] = 0.0;
    for(int i = 0; i < gas_pcnst(); ++i)
      for(int j = 0; j < nqqcwtendaa(); ++j) qqcwgcm_tendaa[i][j] = 0.0;
    for(int n = 0; n <= nsubarea; ++n) {
      if(iscldy_subarea[n]) {
        for(int i = 0; i < gas_pcnst(); ++i)
          qqcwgcm4[i] += qqcwsub4[i][n] * afracsub[n];
        for(int i = 0; i < gas_pcnst(); ++i)
          for(int j = 0; j < nqqcwtendaa(); ++j)
            qqcwgcm_tendaa[i][j] += qqcwsub_tendaa[i][j][n] * afracsub[n];
      }
    }
  }

  for(int lmz = 0; lmz < gas_pcnst(); ++lmz) {
    if(lmapcc_all(lmz) > 0) {
      // HW, to ensure non-negative
      q[lmz] = haero::max(qgcm4[lmz], 0.0);
      if(lmapcc_all(lmz) >= lmapcc_val_aer()) {
        // HW, to ensure non-negative
        qqcw[lmz] = haero::max(qqcwgcm4[lmz], 0.0);
      }
    }
  }
  for(int i = 0; i < gas_pcnst(); ++i) {
    if(iqtend_cond() < nqtendbb())
      q_tendbb[i][iqtend_cond()] = qgcm_tendaa[i][iqtend_cond()];
    if(iqtend_rnam() < nqtendbb())
      q_tendbb[i][iqtend_rnam()] = qgcm_tendaa[i][iqtend_rnam()];
    if(iqtend_nnuc() < nqtendbb())
      q_tendbb[i][iqtend_nnuc()] = qgcm_tendaa[i][iqtend_nnuc()];
    if(iqtend_coag() < nqtendbb())
      q_tendbb[i][iqtend_coag()] = qgcm_tendaa[i][iqtend_coag()];
    if(iqqcwtend_rnam() < nqqcwtendbb())
      qqcw_tendbb[i][iqqcwtend_rnam()] = qqcwgcm_tendaa[i][iqqcwtend_rnam()];
  }
  for(int i = 0; i < num_modes; ++i) qaerwat[i] = qaerwatgcm4[i];
}

}  // namespace scream::impl
