#ifndef EAMXX_MAM_WETSCAV_HPP
#define EAMXX_MAM_WETSCAV_HPP

#include "share/atm_process/atmosphere_process.hpp"
#include "ekat/ekat_parameter_list.hpp"
#include "physics/shoc/shoc_functions.hpp"
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
  using SHF          = shoc::Functions<Real, DefaultDevice>;
  using PF           = scream::PhysicsFunctions<DefaultDevice>;
  using C            = physics::Constants<Real>;
  using KT           = ekat::KokkosTypes<DefaultDevice>;

  using Spack                = typename SHF::Spack;
  using IntSmallPack         = typename SHF::IntSmallPack;
  using Smask                = typename SHF::Smask;
  using view_1d_int          = typename KT::template view_1d<Int>;
  using view_1d              = typename SHF::view_1d<Real>;
  using view_1d_const        = typename SHF::view_1d<const Real>;
  using view_2d              = typename SHF::view_2d<SHF::Spack>;
  using view_2d_const        = typename SHF::view_2d<const Spack>;
  using sview_2d             = typename KokkosTypes<DefaultDevice>::template view_2d<Real>;
  using sview_2d_const       = typename KokkosTypes<DefaultDevice>::template view_2d<const Real>;
  using view_3d              = typename SHF::view_3d<Spack>;
  using view_3d_const        = typename SHF::view_3d<const Spack>;

  using WSM = ekat::WorkspaceManager<Spack, KT::Device>;

  template<typename ScalarT>
  using uview_1d = Unmanaged<typename KT::template view_1d<ScalarT>>;
  template<typename ScalarT>
  using uview_2d = Unmanaged<typename KT::template view_2d<ScalarT>>;

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
  


  // WSM for internal local variables
  ekat::WorkspaceManager<Spack, KT::Device> workspace_mgr;

  std::shared_ptr<const AbstractGrid>   m_grid;
}; // class MAMWetscav

} // namespace scream

#endif // EAMXX_MAM_WETSCAV_HPP
