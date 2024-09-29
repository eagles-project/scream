#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/mam4.hpp>

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

KOKKOS_INLINE_FUNCTION constexpr int maxsubarea() { return 3; }

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

KOKKOS_INLINE_FUNCTION
void mam_amicphys_1gridcell(
    const int kk, const Real (&dgn_a)[AeroConfig::num_modes()],
    const Real (&qsub3)[gas_pcnst()][maxsubarea()],
    const Real (&qqcwsub3)[gas_pcnst()][maxsubarea()],
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
    for(int j = 1; j < maxsubarea(); ++j) {
      qsub4[i][j]    = qsub3[i][j];
      qqcwsub4[i][j] = qqcwsub3[i][j];
    }
  }
  if(kk == 48) {
    printf("mam_amicphys_1gridcell_1b:dgn_a:   %0.15E, %i\n", dgn_a[0], 0);
  }
  for(int i = 0; i < num_modes; ++i) {
    for(int j = 0; j < maxsubarea(); ++j) {
      qaerwatsub4[i][j] = -9999;  // qaerwatsub3[i][j];
    }
  }
  if(kk == 48) {
    printf("mam_amicphys_1gridcell_1c:dgn_a:   %0.15E, %i\n\n\n\n", dgn_a[0],
           0);
  }
}  // mam_amicphys_1gridcell

}  // anonymous namespace

KOKKOS_INLINE_FUNCTION
void modal_aero_amicphys_intr(int kk,
                              const Real dgncur_a[AeroConfig::num_modes()]) {
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

  mam_amicphys_1gridcell(kk, dgn_a, qsub3, qqcwsub3,                  // in
                          qsub4, qqcwsub4, qaerwatsub4);  // inout
}

}  // namespace scream::impl
