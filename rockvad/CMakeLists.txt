add_subdirectory(csrc)

if(ROCKVAD_ENABLE_CPP_API)
  add_subdirectory(cpp-api)
endif()

if(ROCKVAD_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
