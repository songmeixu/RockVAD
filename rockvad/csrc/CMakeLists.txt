include_directories(${CMAKE_SOURCE_DIR})

set(rockvad_core_srcs
  parse-options.cc
  log.cc
  # endpoint.cc
  # features.cc
  # resample.cc
  # greedy-search-decoder.cc
  # hypothesis.cc
  # model.cc
  # decoder.cc
  # recognizer.cc
  # stream.cc
  # symbol-table.cc
  # wave-reader.cc
  # zipformer-model.cc
)
add_library(rockvad-core ${rockvad_core_srcs})

target_link_libraries(rockvad-core PUBLIC
  onnxruntime
  # MNN
  # MNN_Express
  )

if(SHERPA_ENABLE_NPU)
  # message(STATUS "VIP LIBS: ${VIP_LIB_DIR} ${VIP_LIBS}")
  target_link_libraries(rockvad-core PUBLIC
    npulib
    VIPlite
    VIPuser
  )
endif()

if(ROCKVAD_ENABLE_PYTHON AND WIN32)
  install(TARGETS rockvad-core DESTINATION ..)
else()
  install(TARGETS rockvad-core DESTINATION lib)
endif()

if(NOT ROCKVAD_ENABLE_PYTHON)
  if(ROCKVAD_ENABLE_BINARY)
    add_executable(rockvad-onnx rockvad-onnx.cc)
    target_link_libraries(rockvad-onnx PRIVATE rockvad-core)
    install(TARGETS rockvad-onnx DESTINATION bin)

    # if(ROCKVAD_HAS_ALSA)
    #   add_executable(rockvad-alsa rockvad-alsa.cc alsa.cc)
    #   target_link_libraries(rockvad-alsa PRIVATE rockvad-core)

    #   if(DEFINED ENV{ROCKVAD_ALSA_LIB_DIR})
    #     target_link_libraries(rockvad-alsa PRIVATE -L$ENV{ROCKVAD_ALSA_LIB_DIR} -lasound)
    #   else()
    #     target_link_libraries(rockvad-alsa PRIVATE asound)
    #   endif()
    #   install(TARGETS rockvad-alsa DESTINATION bin)
    # endif()

    if(ROCKVAD_ENABLE_PORTAUDIO)
      add_executable(rockvad-microphone
        rockvad-microphone.cc
        microphone.cc
      )

      if(BUILD_SHARED_LIBS)
        set(PA_LIB portaudio)
      else()
        set(PA_LIB portaudio_static)
      endif()

      target_link_libraries(rockvad-microphone PRIVATE ${PA_LIB} rockvad-core)

      install(TARGETS rockvad-microphone DESTINATION bin)
    endif()

    set(hdrs
      wav.h
      # features.h
      # model.h
      # recognizer.h
      # symbol-table.h
      # wave-reader.h
    )

    install(FILES ${hdrs}
      DESTINATION include/rockvad/csrc
    )
  endif()
endif()

if(ROCKVAD_ENABLE_TEST)
  add_executable(test-resample test-resample.cc)
  target_link_libraries(test-resample rockvad-core)
endif()
