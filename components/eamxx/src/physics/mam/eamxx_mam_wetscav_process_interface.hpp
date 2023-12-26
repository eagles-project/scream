#ifndef EAMXX_MAM_WETSCAV_HPP
#define EAMXX_MAM_WETSCAV_HPP

//For MAM4 aerosol configuration
#include <physics/mam/mam_coupling.hpp>

//For declaring wetscav class derived from atm process class
#include "share/atm_process/atmosphere_process.hpp"

#include "ekat/ekat_parameter_list.hpp"
#include "share/util/scream_common_physics_functions.hpp"
#include "share/atm_process/ATMBufferManager.hpp"

#include <string>

namespace scream
{

/*
 * The class responsible to handle the aerosol wetscavenging
 *
 * The AD should store exactly ONE instance of this class stored
 * in its list of subcomponents (the AD should make sure of this).
 *
*/

class MAMWetscav : public scream::AtmosphereProcess
{
 
public:

  // Constructors
  MAMWetscav (const ekat::Comm& comm, const ekat::ParameterList& params);

  // The type of subcomponent
  AtmosphereProcessType type () const { return AtmosphereProcessType::Physics; }

  // The name of the subcomponent
  std::string name () const { return "shoc"; }

  // Set the grid
  void set_grids (const std::shared_ptr<const GridsManager> grids_manager);

#ifndef KOKKOS_ENABLE_CUDA
  // Cuda requires methods enclosing __device__ lambda's to be public
protected:
#endif

  void initialize_impl (const RunType run_type);

protected:

  void run_impl        (const double dt);
  void finalize_impl   ();

  // Keep track of field dimensions and other scalar values
  // needed in shoc_main
  Int m_num_cols;
  Int m_num_levs;
  

  std::shared_ptr<const AbstractGrid>   m_grid;
}; // class MAMWetscav

} // namespace scream

#endif // EAMXX_MAM_WETSCAV_HPP
