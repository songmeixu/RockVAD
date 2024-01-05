#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "rockvad/cpp-api/rockvad.h"
#include "rockvad/csrc/parse-options.h"
#include "rockvad/csrc/wav.h"

static constexpr const char *kUsageMessage = R"(
Voice activity detection with silero-vad (streaming / non-streaming).

Usage:
  ./bin/rockvad-onnx \
    --onnx-model=/path/to/silero_vad.opt.onnx \
    --sample-rate=<sample rate in Hz> \
    --window-frame-ms=<window frame in milliseconds> \
    --threshold=<vad threshold> \
    --min-speech-duration-ms=<minimum speech duration in milliseconds> \
    --min-silence-duration-ms=<minimum silence duration in milliseconds> \
    --speech-pad-ms=<speech padding in milliseconds> \
    --segment-wav=<segment wav true/false> \
    --output-dir=<output directory for segmented wavs> \
    /path/to/foo.wav

Options:
  --onnx-model                Path to the ONNX model file.
  --sample-rate               Sample rate in Hz (default: 16000).
  --window-frame-ms           Window frame size in milliseconds (default: 96).
  --threshold                 VAD threshold, range [0.0, 1.0] (default: 0.5).
  --min-speech-duration-ms    Minimum speech duration to be considered as speech in milliseconds (default: 200).
  --min-silence-duration-ms   Minimum silence duration to be considered as silence in milliseconds (default: 500).
  --speech-pad-ms             Extra padding added to speech segments in milliseconds (default: 300).
  --stream                    Enable streaming mode (default: true).
  --segment-wav               Segment the WAV file (default: false).
  --output-dir                Output directory for segmented WAV files.
)";

int main(int32_t argc, char *argv[]) {
  std::string onnx_model_path;

  // ===== VAD configs =====
  int sample_rate = 16000;
  int window_frame_ms = 96;
  float threshold = 0.5f;
  int min_speech_duration_ms = 200;
  int min_silence_duration_ms = 500;
  int speech_pad_ms = 300;

  // ===== VAD MODE =====
  bool stream = true;
  bool segment_wav = false;
  std::string output_dir = "";

  rockvad::ParseOptions po(kUsageMessage);

  po.Register(
      "onnx-model", &onnx_model_path,
      "onnx model path, e.g. /audio/code/RockVAD/models/silero_vad.opt.onnx");
  po.Register("sample-rate", &sample_rate, "Sample rate in Hz");
  po.Register("window-frame-ms", &window_frame_ms,
              "Window frame size in milliseconds");
  po.Register("threshold", &threshold, "VAD threshold");
  po.Register("min-speech-duration-ms", &min_speech_duration_ms,
              "Minimum speech duration in milliseconds");
  po.Register("min-silence-duration-ms", &min_silence_duration_ms,
              "Minimum silence duration in milliseconds");
  po.Register("speech-pad-ms", &speech_pad_ms,
              "Speech padding in milliseconds");
  po.Register("stream", &stream, "Streaming mode");
  po.Register("segment-wav", &segment_wav, "Segment WAV file");
  po.Register("output-dir", &output_dir,
              "Output directory for segmented WAV files");

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  std::string wav_filename = po.GetArg(1);

  // Read wav
  wav::WavReader wav_reader(wav_filename);
  std::vector<float> input_wav(wav_reader.num_samples());

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    input_wav[i] =
        static_cast<float>(static_cast<int16_t>(*(wav_reader.data() + i))) /
        32768;
  }

  VadIterator vad(onnx_model_path, sample_rate, window_frame_ms, threshold,
                  min_speech_duration_ms, min_silence_duration_ms,
                  speech_pad_ms);

  if (segment_wav) {
    vad.segment_wav(wav_filename, output_dir);
  } else if (0) {
    int window_samples = window_frame_ms * (sample_rate / 1000);

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
    // ==============================================
    // ==== = Example 1 of full function  =====
    // ==============================================
    vad.process(input_wav);

    // 1.a get_speech_timestamps
    std::vector<timestamp_t> stamps = vad.get_speech_timestamps();
    for (int i = 0; i < stamps.size(); i++) {
      std::cout << stamps[i].c_str() << std::endl;
    }

    // 1.b collect_chunks output wav
    std::vector<float> output_wav;
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
