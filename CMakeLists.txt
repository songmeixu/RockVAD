cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(rockvad)

set(ROCKVAD_VERSION "0.0.1")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(ROCKVAD_RPATH_ORIGIN "$ORIGIN")
else()
  set(CMAKE_MACOSX_RPATH ON)
  set(ROCKVAD_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${ROCKVAD_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${ROCKVAD_RPATH_ORIGIN})

# You will find a file compile_commands.json in the build directory
# if it is enabled
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)
option(ROCKVAD_ENABLE_PYTHON "Whether to build Python" OFF)
option(ROCKVAD_ENABLE_PORTAUDIO "Whether to build with portaudio" OFF)
option(ROCKVAD_ENABLE_BINARY "Whether to build the binary rockvad" ON)
option(ROCKVAD_ENABLE_TEST "Whether to build tests" OFF)
option(ROCKVAD_ENABLE_CPP_API "Whether to build C API" ON)
option(ROCKVAD_ENABLE_FFMPEG_EXAMPLES "Whether to enable ffmpeg-examples" OFF)
option(ROCKVAD_ENABLE_NPU "Whether to enable NPU build, on aarch64" OFF)

if(DEFINED ANDROID_ABI)
  message(STATUS "Set ROCKVAD_ENABLE_JNI to ON for Android")
  set(ROCKVAD_ENABLE_JNI ON CACHE BOOL "" FORCE)
endif()

# See
# https://stackoverflow.com/questions/33062728/cmake-link-shared-library-on-windows
if(BUILD_SHARED_LIBS AND MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")
message(STATUS "ROCKVAD_ENABLE_PYTHON ${ROCKVAD_ENABLE_PYTHON}")
message(STATUS "ROCKVAD_ENABLE_PORTAUDIO ${ROCKVAD_ENABLE_PORTAUDIO}")
message(STATUS "ROCKVAD_ENABLE_BINARY ${ROCKVAD_ENABLE_BINARY}")
message(STATUS "ROCKVAD_ENABLE_TEST ${ROCKVAD_ENABLE_TEST}")
message(STATUS "ROCKVAD_ENABLE_CPP_API ${ROCKVAD_ENABLE_CPP_API}")
message(STATUS "ROCKVAD_ENABLE_FFMPEG_EXAMPLES ${ROCKVAD_ENABLE_FFMPEG_EXAMPLES}")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Set the release compiler flags to support debugging
if(ROCKVAD_ENABLE_DEBUG_FOR_RELEASE)
  message(STATUS "Enable debugging for Release")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -g -O0")
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIncludeFileCXX)
check_include_file_cxx(alsa/asoundlib.h ROCKVAD_HAS_ALSA)
if(ROCKVAD_HAS_ALSA)
  add_definitions(-DROCKVAD_ENABLE_ALSA=1)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()

if(WIN32 AND MSVC)
  # disable various warnings for MSVC
  # 4244: 'return': conversion from 'unsigned __int64' to 'int', possible loss of data
  # 4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
  # 4305: 'argument': truncation from 'double' to 'const float'
  # 4334: '<<': result of 32-bit shift implicitly converted to 64 bits
  # 4800: 'int': forcing value to bool 'true' or 'false'
  # 4996: 'fopen': This function or variable may be unsafe
  set(disabled_warnings
      /wd4244
      /wd4267
      /wd4305
      /wd4334
      /wd4800
      /wd4996
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()
endif()

include(onnxruntime)

if(ROCK_ENABLE_NPU)
  include(npu)
endif()

if(ROCKVAD_ENABLE_PORTAUDIO)
  include(portaudio)
endif()

if(ROCKVAD_ENABLE_PYTHON)
  include(pybind11)
endif()

add_subdirectory(rockvad)

if(ROCKVAD_ENABLE_FFMPEG_EXAMPLES)
  add_subdirectory(ffmpeg-examples)
endif()

if(ROCKVAD_ENABLE_CPP_API AND ROCKVAD_ENABLE_BINARY)
  add_subdirectory(cpp-api-examples)
endif()
