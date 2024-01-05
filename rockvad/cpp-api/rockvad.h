#include <limits.h>

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

class timestamp_t {
 public:
  int start;
  int end;

  // default + parameterized constructor
  timestamp_t(int start = -1, int end = -1) : start(start), end(end){};

  // assignment operator modifies object, therefore non-const
  timestamp_t &operator=(const timestamp_t &a) {
    start = a.start;
    end = a.end;
    return *this;
  };

  // equality comparison. doesn't modify object. therefore const.
  bool operator==(const timestamp_t &a) const {
    return (start == a.start && end == a.end);
  };
  std::string c_str() {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "{start:%08d,end:%08d}", start, end);
    return std::string(buf);
  };
};

class VadIterator {
 public:
  // Construction
  VadIterator(const std::string ModelPath, int Sample_rate = 16000,
              int windows_frame_size = 96, float Threshold = 0.3f,
              int min_speech_duration_ms = 250,
              int min_silence_duration_ms = 100, int speech_pad_ms = 30,
              int max_speech_duration_s = INT_MAX);

  void reset_states();

  float forward_chunk(const std::vector<float> &data_chunk);

  void stream_predict(const std::vector<float> &data);

  void process(const std::vector<float> &input_wav);

  void process(const std::vector<float> &input_wav,
               std::vector<float> &output_wav);

  void collect_chunks(const std::vector<float> &input_wav,
                      std::vector<float> &output_wav);

  const std::vector<timestamp_t> get_speech_timestamps() const {
    return speeches;
  }

  void drop_chunks(const std::vector<float> &input_wav,
                   std::vector<float> &output_wav);

  // compare to stream_predict(), this:
  //    - support speech_pad_samples
  //    - use uint instead of timestamp
  //    - and return seconds
  void stream_predict2(const std::vector<float> &data,
                       bool return_seconds = true);

  void segment_wav(const std::string &wav_path,
                   const std::string &output_dir = "",
                   bool return_seconds = true);

 private:
  // model config
  int window_size_samples;  // Assign when init, support 256 512 768 for 8k;
                            // 512 1024 1536 for 16k.
  int sample_rate;          // Assign when init support 16000 or 8000
  int sr_per_ms;            // Assign when init, support 8 or 16
  float threshold;

  int min_silence_samples;
  int min_silence_samples_at_max_speech;  // sr_per_ms * #98
  int min_speech_samples;                 // sr_per_ms * #ms
  float max_speech_samples;
  int speech_pad_samples;
  int audio_length_samples;

  // model states
  bool triggered = false;
  unsigned int speech_start = 0;
  unsigned int speech_end = 0;
  unsigned int temp_end =
      0;  // to save potential segment end (and tolerate some silence)
  unsigned int current_sample = 0;
  // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes

  int prev_end;
  int next_start = 0;

  // Output timestamp
  std::vector<timestamp_t> speeches;
  timestamp_t current_speech;

  // Call it in predict func. if you prefer raw bytes input.
  void bytes_to_float_tensor(const char *pcm_bytes);

  // OnnxRuntime
  void init_engine_threads(int inter_threads, int intra_threads);

  void init_onnx_model(const std::string &model_path);
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::shared_ptr<Ort::Session> session = nullptr;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

  // Onnx model
  // Inputs
  std::vector<Ort::Value> ort_inputs;

  std::vector<const char *> input_node_names = {"input", "sr", "h", "c"};
  std::vector<float> input;
  std::vector<int64_t> sr;
  unsigned int size_hc = 2 * 1 * 64;  // It's FIXED.
  std::vector<float> _h;
  std::vector<float> _c;

  int64_t input_node_dims[2] = {};
  const int64_t sr_node_dims[1] = {1};
  const int64_t hc_node_dims[3] = {2, 1, 64};

  // Outputs
  std::vector<Ort::Value> ort_outputs;
  std::vector<const char *> output_node_names = {"output", "hn", "cn"};
};