#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "rockvad/cpp-api/rockvad.h"
#include "rockvad/csrc/parse-options.h"
#include "rockvad/csrc/wav.h"

static constexpr const char *kUsageMessage = R"(
Online (streaming) voice activity detection with rockvad.

Usage:
  ./bin/rockvad-onnx \
    --onnx-model=/path/to/silero_vad.opt.onnx \
    /path/to/foo.wav
)";

int main(int32_t argc, char *argv[]) {
  std::string onnx_model_path;
  bool stream = true;
  bool segment_wav = false;
  float threshold = 0.3f;

  rockvad::ParseOptions po(kUsageMessage);

  po.Register(
      "onnx-model", &onnx_model_path,
      "onnx model path, e.g. /audio/code/RockVAD/models/silero_vad.opt.onnx");
  po.Register("stream", &stream, "streaming mode");
  po.Register("segment-wav", &segment_wav, "segment wav file");
  po.Register("threshold", &threshold, "threshold for vad");


  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  std::string wav_filename = po.GetArg(1);

  // Read wav
  wav::WavReader wav_reader(wav_filename);
  std::vector<int16_t> data(wav_reader.num_samples());
  std::vector<float> input_wav(wav_reader.num_samples());

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    data[i] = static_cast<int16_t>(*(wav_reader.data() + i));
  }

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    input_wav[i] = static_cast<float>(data[i]) / 32768;
  }

  // ===== Test configs =====

  int test_sr = 16000;
  int test_frame_ms = 96;
  float test_threshold = threshold;
  int test_min_speech_duration_ms = 3000;
  int test_min_silence_duration_ms = 2500;
  int test_speech_pad_ms = 30;
  int test_window_samples = test_frame_ms * (test_sr / 1000);

  VadIterator vad(onnx_model_path, test_sr, test_frame_ms, test_threshold,
                  test_min_speech_duration_ms, test_min_silence_duration_ms,
                  test_speech_pad_ms);

  if (segment_wav) {
    vad.segment_wav(wav_filename);
  } else {
    for (int j = 0; j < wav_reader.num_samples(); j += test_window_samples) {
      std::vector<float> r{&input_wav[0] + j,
                           &input_wav[0] + j + test_window_samples};

      // auto start = std::chrono::high_resolution_clock::now();

      // Predict and print throughout process time
      vad.stream_predict2(r);

      // auto end = std::chrono::high_resolution_clock::now();
      // auto elapsed_time =
      //     std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      // std::cout << "== Elapsed time: " << 1.0 * elapsed_time.count() /
      // 1000000
      //           << "ms"
      //           << " ==" << std::endl;
    }
  }

  return 0;
}
