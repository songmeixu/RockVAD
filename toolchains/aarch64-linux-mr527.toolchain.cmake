set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# the toolchain path is from mr527 docker image, could be changed to your own toolchain path
# set(CMAKE_C_COMPILER "/root/toolchain-aarch64_generic-glibc-1130/bin/aarch64-openwrt-linux-gcc")
# set(CMAKE_CXX_COMPILER "/root/toolchain-aarch64_generic-glibc-1130/bin/aarch64-openwrt-linux-g++")
set(CMAKE_C_COMPILER "aarch64-openwrt-linux-gcc")
set(CMAKE_CXX_COMPILER "aarch64-openwrt-linux-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=armv8.2-a+fp16")
set(CMAKE_CXX_FLAGS "-march=armv8.2-a+fp16")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
