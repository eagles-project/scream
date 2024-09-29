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
KOKKOS_INLINE_FUNCTION constexpr int iqtend_cond() { return 0; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_rnam() { return 1; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_nnuc() { return 2; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_coag() { return 3; }
KOKKOS_INLINE_FUNCTION constexpr int iqtend_cond_only() { return 4; }
KOKKOS_INLINE_FUNCTION constexpr int iqqcwtend_rnam() { return 0; }
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

// Returns index of aerosol numbers in gas_pcnst array
KOKKOS_INLINE_FUNCTION int lmap_numcw(const int mode) {
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

KOKKOS_INLINE_FUNCTION int lmap_aercw(const int iaer, const int mode) {
  return lmap_aer(iaer, mode);
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

KOKKOS_INLINE_FUNCTION
void mam_amicphys_1gridcell(
    const int kk, const Real (&dgn_a)[AeroConfig::num_modes()],
    const Real (&qsub3)[gas_pcnst()][maxsubarea()],
    const Real (&qqcwsub3)[gas_pcnst()][maxsubarea()],
    const Real (&qaerwatsub3)[AeroConfig::num_modes()][maxsubarea()],
    Real (&qsub4)[gas_pcnst()][maxsubarea()],
    Real (&qqcwsub4)[gas_pcnst()][maxsubarea()],
    Real (&qaerwatsub4)[AeroConfig::num_modes()][maxsubarea()]) {
  if(kk == 48) {
    printf("mam_amicphys_1gridcell:dgn_a:   %0.15E, %i\n", dgn_a[0], 0);
  }
  static constexpr int num_gas_ids = AeroConfig::num_gas_ids();
  static constexpr int num_modes   = AeroConfig::num_modes();

  if(kk == 48) {
    printf("mam_amicphys_1gridcell_1a:dgn_a:   %0.15E, %i\n", dgn_a[0], 0);
  }

  for(int i = 0; i < num_gas_ids; ++i) {
    for(int j = 1; j <= maxsubarea(); ++j) {
      qsub4[i][j]    = qsub3[i][j];
      qqcwsub4[i][j] = qqcwsub3[i][j];
    }
  }
  if(kk == 48) {
    printf("mam_amicphys_1gridcell_1b:dgn_a:   %0.15E, %i\n", dgn_a[0], 0);
  }
  for(int i = 0; i < num_modes; ++i) {
    for(int j = 0; j <= maxsubarea(); ++j) {
      qaerwatsub4[i][j] = 0;  // qaerwatsub3[i][j];
    }
  }
  if(kk == 48) {
    printf("mam_amicphys_1gridcell_1c:dgn_a:   %0.15E, %i\n\n\n\n", dgn_a[0],
           0);
  }
}  // mam_amicphys_1gridcell

}  // anonymous namespace

KOKKOS_INLINE_FUNCTION
void modal_aero_amicphys_intr(
    int kk, const Real dgncur_a[AeroConfig::num_modes()],
    const Real dgncur_awet[AeroConfig::num_modes()],
    const Real wetdens_host[AeroConfig::num_modes()]) {
  static constexpr int num_modes = AeroConfig::num_modes();

  Real qaerwatsub3[num_modes][maxsubarea()];
  Real qsub3[gas_pcnst()][maxsubarea()];

  Real qqcwsub3[gas_pcnst()][maxsubarea()];

  assign_2d_array(num_modes, maxsubarea(), 0.0,  // in
                  qaerwatsub3);                  // out

  assign_2d_array(gas_pcnst(), maxsubarea(), 0.0,  // in
                  qsub3);                          // out
  assign_2d_array(gas_pcnst(), maxsubarea(), 0.0,  // in
                  qqcwsub3);                       // out

  if(kk == 48) {
    for(int icnst = 0; icnst < gas_pcnst(); ++icnst) {
      // printf("lmapcc_aa:%i, %i\n", lmapcc_all(icnst), icnst);
    }
  }

  //  Initialize the "after-amicphys" values
  Real qsub4[gas_pcnst()][maxsubarea()]     = {};
  Real qqcwsub4[gas_pcnst()][maxsubarea()]  = {};
  Real qaerwatsub4[num_modes][maxsubarea()] = {};

  Real dgn_a[num_modes];
  assign_1d_array(num_modes, 0.0,  // in
                  dgn_a);          // out

  for(int n = 0; n < num_modes; ++n) {
    dgn_a[n] = dgncur_a[n];
    if(kk == 48) {
      printf("dgn_a:%0.15E, %i\n", dgn_a[n], n);
    }
  }

  mam_amicphys_1gridcell(
      kk,
      // FIXME: dgn_a, dgn_awet, wetdens seems like "in", confirm it
      dgn_a, qsub3, qqcwsub3,                      // in
      qaerwatsub3, qsub4, qqcwsub4, qaerwatsub4);  // inout
}

}  // namespace scream::impl
