function(download_onnxruntime)
  include(FetchContent)

  # Please also change ../pack-for-embedded-systems.sh
  # onnxruntime v1.15.1
  set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz")
  set(onnxruntime_HASH "SHA256=5492f9065f87538a286fb04c8542e9ff7950abb2ea6f8c24993a940006787d87")
  # onnxruntime v1.12.1
  # set(onnxruntime_URL  "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz")
  # set(onnxruntime_HASH "SHA256=8f6eb9e2da9cf74e7905bf3fc687ef52e34cc566af7af2f92dafe5a5d106aa3d")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  set(possible_file_locations
    $ENV{HOME}/Downloads/onnxruntime-linux-x64-1.15.1.tgz
    ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.15.1.tgz
    ${PROJECT_BINARY_DIR}/onnxruntime-linux-x64-1.15.1.tgz
    /tmp/onnxruntime-linux-x64-1.15.1.tgz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(onnxruntime_URL  "${f}")
      file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
      break()
    endif()
  endforeach()

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
