include_directories(${CMAKE_SOURCE_DIR})

add_executable(detect-file-cpp-api detect-file-cpp-api.cc)
target_link_libraries(detect-file-cpp-api rockvad-cpp-api)
