include_directories(${CMAKE_SOURCE_DIR})
add_executable(detect-file-c-api detect-file-c-api.c)
target_link_libraries(detect-file-c-api rockvad-c-api)
