#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "rockvad/cpp-api/rockvad.h"
#include "rockvad/csrc/wav.h"

int main() {
  // Read wav
  // wav::WavReader wav_reader("../../en_example.wav");
  wav::WavReader wav_reader("/audio/data/roborock/nihao_shitou/wav/smx/46.wav");
  std::vector<int16_t> data(wav_reader.num_samples());
  std::vector<float> input_wav(wav_reader.num_samples());

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    data[i] = static_cast<int16_t>(*(wav_reader.data() + i));
  }

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    input_wav[i] = static_cast<float>(data[i]) / 32768;
  }

  // ===== Test configs =====
  std::string path = "/audio/code/RockVAD/models/silero_vad.opt.onnx";
  int test_sr = 16000;
  int test_frame_ms = 96;
  float test_threshold = 0.5f;
  int test_min_silence_duration_ms = 0;
  int test_speech_pad_ms = 0;
  int test_window_samples = test_frame_ms * (test_sr / 1000);

  VadIterator vad(path, test_sr, test_frame_ms, test_threshold,
                  test_min_silence_duration_ms, test_speech_pad_ms);

  for (int j = 0; j < wav_reader.num_samples(); j += test_window_samples) {
    // std::cout << "== 4" << std::endl;
    std::vector<float> r{&input_wav[0] + j,
                         &input_wav[0] + j + test_window_samples};
    auto start = std::chrono::high_resolution_clock::now();
    // Predict and print throughout process time
    vad.stream_predict(r);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "== Elapsed time: " << 1.0 * elapsed_time.count() / 1000000
              << "ms"
              << " ==" << std::endl;
  }
}
