add_subdirectory(csrc)

if(ROCKVAD_ENABLE_C_API)
  add_subdirectory(c-api)
endif()

if(ROCKVAD_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
