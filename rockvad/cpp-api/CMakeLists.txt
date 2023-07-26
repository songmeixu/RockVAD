include_directories(${CMAKE_SOURCE_DIR})
add_library(rockvad-cpp-api rockvad.cc)
target_link_libraries(rockvad-cpp-api rockvad-core)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(rockvad-cpp-api PRIVATE ROCKVAD_BUILD_SHARED_LIBS=1)
  target_compile_definitions(rockvad-cpp-api PRIVATE ROCKVAD_BUILD_MAIN_LIB=1)
endif()

install(TARGETS rockvad-cpp-api DESTINATION lib)

install(FILES rockvad.h
  DESTINATION include/rockvad/cpp-api
)
