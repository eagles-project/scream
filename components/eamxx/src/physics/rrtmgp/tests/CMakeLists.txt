if (SCREAM_ONLY_GENERATE_BASELINES)
  # Build baseline code
  add_executable(generate_baseline generate_baseline.cpp)
  target_link_libraries(generate_baseline PUBLIC scream_rrtmgp rrtmgp_test_utils)

  # Generate allsky baseline with the usual cmake custom command-target pair pattern
  # Note: these "baselines" are not to compare scream with a previous version, but
  #       rather to compare scream::rrtmgp with raw rrtmgp.
  CreateUnitTestFromExec(
    rrtmgp-allsky-baseline generate_baseline
    LABELS baseline_gen rrtmgp
    EXE_ARGS "${SCREAM_DATA_DIR}/init/rrtmgp-allsky.nc ${SCREAM_BASELINES_DIR}/data/rrtmgp-allsky-baseline.nc"
  )

else ()

  if (SCREAM_ENABLE_BASELINE_TESTS)
    # NOTE: tests inside this branch won't be built in a baselines-only build
    CreateUnitTest(rrtmgp_tests rrtmgp_tests.cpp
        LIBS scream_rrtmgp rrtmgp_test_utils
        LABELS "rrtmgp;physics"
        EXE_ARGS "-i ${SCREAM_DATA_DIR}/init/rrtmgp-allsky.nc -b ${SCREAM_BASELINES_DIR}/data/rrtmgp-allsky-baseline.nc"
        EXCLUDE_MAIN_CPP
    )
  endif()

  CreateUnitTest(rrtmgp_unit_tests rrtmgp_unit_tests.cpp
      LIBS scream_rrtmgp rrtmgp_test_utils
      LABELS "rrtmgp;physics"
  )
endif()
