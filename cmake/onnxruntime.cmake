function(download_onnxruntime)
  include(FetchContent)

  # Please also change ../pack-for-embedded-systems.sh\
  set(ONNX_VERSION "1.15.1")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(onnxruntime_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
    set(onnxruntime_HASH "SHA256=85272e75d8dd841138de4b774a9672ea93c1be108d96038c6c34a62d7f976aee")
  else()
    set(onnxruntime_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
    set(onnxruntime_HASH "SHA256=5492f9065f87538a286fb04c8542e9ff7950abb2ea6f8c24993a940006787d87")
  endif()

  FetchContent_Declare(onnxruntime
    URL      ${onnxruntime_URL}
    URL_HASH ${onnxruntime_HASH}
  )

  FetchContent_MakeAvailable(onnxruntime)

  message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")
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
