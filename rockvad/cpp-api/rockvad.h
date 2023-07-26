#include <vector>

#include "onnxruntime_cxx_api.h"

class VadIterator {
 public:
  // Construction
  VadIterator(const std::string ModelPath, int Sample_rate, int frame_size,
              float Threshold, int min_silence_duration_ms, int speech_pad_ms);

  void reset_states();

  void predict(const std::vector<float> &data);

 private:
  // model config
  int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k;
                                // 512 1024 1536 for 16k.
  int sample_rate;
  int sr_per_ms;  // Assign when init, support 8 or 16
  float threshold;
  int min_silence_samples;  // sr_per_ms * #ms
  int speech_pad_samples;   // usually a

  // model states
  bool triggerd = false;
  unsigned int speech_start = 0;
  unsigned int speech_end = 0;
  unsigned int temp_end = 0;
  unsigned int current_sample = 0;
  // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
  float output;

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

 public:
};