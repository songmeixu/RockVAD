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

  // ===== VAD configs =====
  int sample_rate = 16000;
  int window_frame_ms = 96;
  float threshold = 0.5f;
  ;
  int min_speech_duration_ms = 200;
  int min_silence_duration_ms = 500;
  int speech_pad_ms = 300;

  // ===== VAD MODE =====
  bool stream = true;
  bool segment_wav = false;

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
  std::vector<float> output_wav;

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    data[i] = static_cast<int16_t>(*(wav_reader.data() + i));
  }

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    input_wav[i] = static_cast<float>(data[i]) / 32768;
  }

  int window_samples = window_frame_ms * (sample_rate / 1000);

  VadIterator vad(onnx_model_path, sample_rate, window_frame_ms, threshold,
                  min_speech_duration_ms, min_silence_duration_ms,
                  speech_pad_ms);

  if (segment_wav) {
    vad.segment_wav(wav_filename);
  } else if (0) {
    for (int j = 0; j < wav_reader.num_samples(); j += window_samples) {
      std::vector<float> r{&input_wav[0] + j,
                           &input_wav[0] + j + window_samples};

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
  } else {
    std::vector<timestamp_t> stamps;

    // ==============================================
    // ==== = Example 1 of full function  =====
    // ==============================================
    vad.process(input_wav);

    // 1.a get_speech_timestamps
    stamps = vad.get_speech_timestamps();
    for (int i = 0; i < stamps.size(); i++) {
      std::cout << stamps[i].c_str() << std::endl;
    }

    // 1.b collect_chunks output wav
    vad.collect_chunks(input_wav, output_wav);

    // 1.c drop_chunks output wav
    vad.drop_chunks(input_wav, output_wav);

    // ==============================================
    // ===== Example 2 of simple full function  =====
    // ==============================================
    vad.process(input_wav, output_wav);

    stamps = vad.get_speech_timestamps();
    for (int i = 0; i < stamps.size(); i++) {
      std::cout << stamps[i].c_str() << std::endl;
    }

    // ==============================================
    // ===== Example 3 of full function  =====
    // ==============================================
    for (int i = 0; i < 2; i++) vad.process(input_wav, output_wav);
  }

  return 0;
}
