include_directories(${CMAKE_SOURCE_DIR})
add_library(rockvad-c-api rockvad.cc)
target_link_libraries(rockvad-c-api rockvad-core)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(rockvad-c-api PRIVATE ROCKVAD_BUILD_SHARED_LIBS=1)
  target_compile_definitions(rockvad-c-api PRIVATE ROCKVAD_BUILD_MAIN_LIB=1)
endif()

install(TARGETS rockvad-c-api DESTINATION lib)

install(FILES rockvad.h
  DESTINATION include/rockvad/c-api
)
