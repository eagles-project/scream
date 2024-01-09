#include <catch2/catch.hpp>

#include "control/atmosphere_driver.hpp"
#include "diagnostics/register_diagnostics.hpp"

#include "physics/mam/eamxx_mam_wetscav_process_interface.hpp"

#include "share/grid/mesh_free_grids_manager.hpp"
#include "share/atm_process/atmosphere_process.hpp"

#include "ekat/ekat_parse_yaml_file.hpp"

#include <iomanip>

namespace scream {

TEST_CASE("mam_wetscav-stand-alone", "") {
  using namespace scream;
  using namespace scream::control;

  // Create a comm
  ekat::Comm atm_comm (MPI_COMM_WORLD);

  // Load ad parameter list
  std::string fname = "input_mam_wetscav.yaml";
  ekat::ParameterList ad_params("Atmosphere Driver");
  parse_yaml_file(fname,ad_params) ;

  // Time stepping parameters
  const auto& ts     = ad_params.sublist("time_stepping");
  const auto  dt     = ts.get<int>("time_step");
  const auto  nsteps = ts.get<int>("number_of_steps");
  const auto  t0_str = ts.get<std::string>("run_t0");
  const auto  t0     = util::str_to_time_stamp(t0_str);

  EKAT_ASSERT_MSG (dt>0, "Error! Time step must be positive.\n");

  // Need to register products in the factory *before* we create any atm process or grids manager.
  auto& proc_factory = AtmosphereProcessFactory::instance();
  auto& gm_factory = GridsManagerFactory::instance();
  proc_factory.register_product("mam_wetscav",&create_atmosphere_process<MAMWetscav>);
  gm_factory.register_product("Mesh Free",&create_mesh_free_grids_manager);
  register_diagnostics();

  // Create the driver
  AtmosphereDriver ad;

  // Init and run
  ad.initialize(atm_comm,ad_params,t0);

  if (atm_comm.am_i_root()) {
    printf("Start time stepping loop...       [  0%%]\n");
  }
  // MUST FIXME: change 1 to nsteps in the following loop
  for (int i=0; i<1; ++i) {
    ad.run(dt);
    if (atm_comm.am_i_root()) {
      std::cout << "  - Iteration " << std::setfill(' ') << std::setw(3) << i+1 << " completed";
      std::cout << "       [" << std::setfill(' ') << std::setw(3) << 100*(i+1)/nsteps << "%]\n";
    }
  }

  // Finalize
  ad.finalize();

  // If we got here, we were able to run mam wetscav
  REQUIRE(true);
}

} // empty namespace
