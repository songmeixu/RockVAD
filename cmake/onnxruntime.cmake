function(download_onnxruntime)
  include(FetchContent)

  set(ONNX_VERSION "1.15.1")

  # Check if the system processor is aarch64 (ARM64)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Check if the system name is Darwin (macOS/OSX)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
      # macOS/OSX arm64 specific settings
      set(onnxruntime_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz")
      set(onnxruntime_HASH "SHA256=df97832fc7907c6677a6da437f92339d84a462becb74b1d65217fcb859ee9460")
    else()
      # Linux ARM64 settings
      set(onnxruntime_URL "${PROJECT_SOURCE_DIR}/cmake/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
      set(onnxruntime_HASH "SHA256=aa938bb9211d28080b3e8f5e25b50b87257ec201301b491f5296595a9b755392")
    endif()
  else()
    # General settings for systems other than ARM64
    set(onnxruntime_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
    set(onnxruntime_HASH "SHA256=5492f9065f87538a286fb04c8542e9ff7950abb2ea6f8c24993a940006787d87")
  endif()

  FetchContent_Declare(onnxruntime
    URL      ${onnxruntime_URL}
    URL_HASH ${onnxruntime_HASH}
  )

  FetchContent_MakeAvailable(onnxruntime)

  message(STATUS "onnxruntime (${onnxruntime_URL}) is downloaded to ${onnxruntime_SOURCE_DIR}")
  message(STATUS "onnxruntime's binary dir is ${onnxruntime_BINARY_DIR}")

  # add_subdirectory(${onnxruntime_SOURCE_DIR} ${onnxruntime_BINARY_DIR})
  include_directories(${onnxruntime_SOURCE_DIR}/include)
  link_directories(${onnxruntime_SOURCE_DIR}/lib)

  # if(ROCKVAD_ENABLE_PYTHON AND WIN32)
  #   install(TARGETS onnxruntime DESTINATION ..)
  # else()
  #   install(TARGETS onnxruntime DESTINATION lib)
  # endif()
endfunction()

download_onnxruntime()
