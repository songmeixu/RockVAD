#include "rockvad/cpp-api/rockvad.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>  // 需要C++17或更高版本
#include <iostream>
#include <sstream>
#include <string>

#include "rockvad/csrc/wav.h"

VadIterator::VadIterator(const std::string ModelPath, int Sample_rate,
                         int windows_frame_size, float Threshold,
                         int min_speech_duration_ms,
                         int min_silence_duration_ms, int speech_pad_ms,
                         int max_speech_duration_s) {
  init_onnx_model(ModelPath);
  sample_rate = Sample_rate;
  sr_per_ms = sample_rate / 1000;
  threshold = Threshold;

  min_speech_samples = sr_per_ms * min_speech_duration_ms;
  max_speech_samples = max_speech_duration_s == INT_MAX
                           ? INT_MAX
                           : sample_rate * max_speech_duration_s -
                                 window_size_samples - 2 * speech_pad_samples;
  min_silence_samples = sr_per_ms * min_silence_duration_ms;
  min_silence_samples_at_max_speech = sr_per_ms * 98;
  speech_pad_samples = sr_per_ms * speech_pad_ms;
  window_size_samples = sr_per_ms * windows_frame_size;

  input.resize(window_size_samples);
  input_node_dims[0] = 1;
  input_node_dims[1] = window_size_samples;

  _h.resize(size_hc);
  _c.resize(size_hc);
  sr.resize(1);
  sr[0] = sample_rate;
}

void VadIterator::init_onnx_model(const std::string &model_path) {
  // Init threads = 1 for
  init_engine_threads(1, 1);
  // Load model
  session =
      std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
}

void VadIterator::init_engine_threads(int inter_threads, int intra_threads) {
  // The method should be called in each thread/proc in multi-thread/proc work
  session_options.SetIntraOpNumThreads(intra_threads);
  session_options.SetInterOpNumThreads(inter_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void VadIterator::reset_states() {
  // Call reset before each audio start
  std::memset(_h.data(), 0.0f, _h.size() * sizeof(float));
  std::memset(_c.data(), 0.0f, _c.size() * sizeof(float));
  triggered = false;
  temp_end = 0;
  current_sample = 0;

  prev_end = next_start = 0;

  speeches.clear();
  current_speech = timestamp_t();
}

// Call it in predict func. if you prefer raw bytes input.
void VadIterator::bytes_to_float_tensor(const char *pcm_bytes) {
  const int16_t *in_data = reinterpret_cast<const int16_t *>(pcm_bytes);
  for (int i = 0; i < window_size_samples; i++) {
    input[i] =
        static_cast<float>(in_data[i]) / 32768;  // int16_t normalized to float
  }
}

float VadIterator::forward_chunk(const std::vector<float> &data_chunk) {
  // Infer
  // Create ort tensors
  input.assign(data_chunk.begin(), data_chunk.end());
  Ort::Value input_ort = Ort::Value::CreateTensor<float>(
      memory_info, input.data(), input.size(), input_node_dims, 2);
  Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, sr.data(), sr.size(), sr_node_dims, 1);
  Ort::Value h_ort = Ort::Value::CreateTensor<float>(
      memory_info, _h.data(), _h.size(), hc_node_dims, 3);
  Ort::Value c_ort = Ort::Value::CreateTensor<float>(
      memory_info, _c.data(), _c.size(), hc_node_dims, 3);

  // Clear and add inputs
  ort_inputs.clear();
  ort_inputs.emplace_back(std::move(input_ort));
  ort_inputs.emplace_back(std::move(sr_ort));
  ort_inputs.emplace_back(std::move(h_ort));
  ort_inputs.emplace_back(std::move(c_ort));

  // Infer
  ort_outputs = session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
      ort_inputs.size(), output_node_names.data(), output_node_names.size());

  // Output probability & update h,c recursively
  float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
  float *hn = ort_outputs[1].GetTensorMutableData<float>();
  std::memcpy(_h.data(), hn, size_hc * sizeof(float));
  float *cn = ort_outputs[2].GetTensorMutableData<float>();
  std::memcpy(_c.data(), cn, size_hc * sizeof(float));

  return speech_prob;
}

void VadIterator::stream_predict(const std::vector<float> &data) {
  // Output probability
  float speech_prob = forward_chunk(data);

  // Push forward sample index
  current_sample += window_size_samples;

  // Reset temp_end when > threshold
  if ((speech_prob >= threshold)) {
    if (temp_end != 0) {
      temp_end = 0;
      if (next_start < prev_end)
        next_start = current_sample - window_size_samples;
    }

    // 1) Start
    if (triggered == false) {
      triggered = true;

      current_speech.start = current_sample - window_size_samples;
    }
    return;
  }

  // 2) Speak too long
  if ((triggered == true) &&
      ((current_sample - current_speech.start) > max_speech_samples)) {
    if (prev_end > 0) {
      current_speech.end = prev_end;
      speeches.push_back(current_speech);
      current_speech = timestamp_t();

      // previously reached silence(< neg_thres) and is still not speech(<
      // thres)
      if (next_start < prev_end)
        triggered = false;
      else {
        current_speech.start = next_start;
      }
      prev_end = 0;
      next_start = 0;
      temp_end = 0;

    } else {
      current_speech.end = current_sample;
      speeches.push_back(current_speech);
      current_speech = timestamp_t();
      prev_end = 0;
      next_start = 0;
      temp_end = 0;
      triggered = false;
    }
    return;
  }

  // 3) Keep going (speaking or silence)
  if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold)) {
    return;
  }

  if ((speech_prob < (threshold - 0.15))) {
    // 4) End
    if (triggered == true) {
      if (temp_end == 0) {
        temp_end = current_sample;
      }
      if (current_sample - temp_end > min_silence_samples_at_max_speech)
        prev_end = temp_end;
      // a. silence < min_slience_samples, continue speaking
      if ((current_sample - temp_end) < min_silence_samples) {
      }
      // b. silence >= min_slience_samples, end speaking
      else {
        current_speech.end = temp_end;
        if (current_speech.end - current_speech.start > min_speech_samples) {
          speeches.push_back(current_speech);
          current_speech = timestamp_t();
          prev_end = 0;
          next_start = 0;
          temp_end = 0;
          triggered = false;
        }
      }
    } else {
      // may first windows see end state.
    }

    return;
  }
}

void VadIterator::process(const std::vector<float> &input_wav) {
  reset_states();

  audio_length_samples = input_wav.size();

  for (int j = 0; j < audio_length_samples; j += window_size_samples) {
    if (j + window_size_samples > audio_length_samples) break;
    std::vector<float> r{&input_wav[0] + j,
                         &input_wav[0] + j + window_size_samples};
    stream_predict(r);
  }

  if (current_speech.start >= 0) {
    current_speech.end = audio_length_samples;
    speeches.push_back(current_speech);
    current_speech = timestamp_t();
    prev_end = 0;
    next_start = 0;
    temp_end = 0;
    triggered = false;
  }
};

void VadIterator::process(const std::vector<float> &input_wav,
                          std::vector<float> &output_wav) {
  process(input_wav);
  collect_chunks(input_wav, output_wav);
}

void VadIterator::collect_chunks(const std::vector<float> &input_wav,
                                 std::vector<float> &output_wav) {
  output_wav.clear();
  for (int i = 0; i < speeches.size(); i++) {
    std::vector<float> slice(&input_wav[speeches[i].start],
                             &input_wav[speeches[i].end]);
    output_wav.insert(output_wav.end(), slice.begin(), slice.end());
  }
};

void VadIterator::drop_chunks(const std::vector<float> &input_wav,
                              std::vector<float> &output_wav) {
  output_wav.clear();
  int current_start = 0;
  for (int i = 0; i < speeches.size(); i++) {
    std::vector<float> slice(&input_wav[current_start],
                             &input_wav[speeches[i].start]);
    output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    current_start = speeches[i].end;
  }

  std::vector<float> slice(&input_wav[current_start],
                           &input_wav[input_wav.size()]);
  output_wav.insert(output_wav.end(), slice.begin(), slice.end());
};

void VadIterator::stream_predict2(const std::vector<float> &data,
                                  bool return_seconds) {
  // Output probability
  float speech_prob = forward_chunk(data);

  // Push forward sample index
  current_sample += window_size_samples;

  // Reset temp_end when > threshold
  if ((speech_prob >= threshold) && (temp_end != 0)) {
    temp_end = 0;
  }
  // 1) Silence
  if ((speech_prob < threshold) && (triggered == false)) {
    // printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
  }
  // 2) Speaking
  if ((speech_prob >= (threshold - 0.15)) && (triggered == true)) {
    // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample / sample_rate);
  }

  // 3) Start
  if ((speech_prob >= threshold) && (triggered == false)) {
    triggered = true;
    speech_start = current_sample - window_size_samples -
                   speech_pad_samples;  // minus window_size_samples to get
                                        // precise start time point.
  }

  // 4) End
  if ((speech_prob < (threshold - 0.15)) && (triggered == true)) {
    if (temp_end == 0) {
      temp_end = current_sample;
    }
    // a. silence < min_slience_samples, continue speaking
    if ((current_sample - temp_end) < min_silence_samples) {
      // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample /
      // sample_rate); printf("");
    }
    // b. silence >= min_slience_samples, end speaking
    else {
      speech_end = temp_end ? temp_end + speech_pad_samples
                            : current_sample + speech_pad_samples;
      if (speech_end - speech_start > min_speech_samples) {
        if (return_seconds) {
          printf("{ start: %.3f s }\n", 1.0 * speech_start / sample_rate);
          printf("{ end: %.3f s }\n", 1.0 * speech_end / sample_rate);
        } else {
          printf("{ start: %d }\n", speech_start);
          printf("{ end: %d }\n", speech_end);
        }
      }
      temp_end = 0;
      triggered = false;
    }
  }
}

void VadIterator::segment_wav(const std::string &wav_path,
                              const std::string &output_dir,
                              bool return_seconds) {
  // Read wav
  wav::WavReader wav_reader(wav_path);
  std::vector<int16_t> data(wav_reader.num_samples());
  std::vector<float> input_wav(wav_reader.num_samples());

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    data[i] = static_cast<int16_t>(*(wav_reader.data() + i));
  }

  for (int i = 0; i < wav_reader.num_samples(); i++) {
    input_wav[i] = static_cast<float>(data[i]) / 32768;
  }

  std::vector<float> speech_probs;
  for (int i = 0; i < input_wav.size(); i += window_size_samples) {
    // Calculate the actual size of the data_chunk (remaining samples)
    int chunk_size =
        std::min(window_size_samples, static_cast<int>(input_wav.size()) - i);

    // Create a vector with window_size_samples size and initialize with zeros
    std::vector<float> data_chunk(window_size_samples, 0.0f);

    // Copy the actual data from input_wav to data_chunk
    std::copy_n(input_wav.begin() + i, chunk_size, data_chunk.begin());

    speech_probs.push_back(forward_chunk(data_chunk));
  }

  std::vector<std::pair<float, float>> speeches;
  std::pair<float, float> current_speech = std::make_pair(-1.0f, -1.0f);

  for (int i = 0; i < speech_probs.size(); i++) {
    // Push forward sample index
    current_sample = window_size_samples * (i + 1);

    float speech_prob = speech_probs[i];

    // Reset temp_end when > threshold
    if ((speech_prob >= threshold) && (temp_end != 0)) {
      temp_end = 0;
    }
    // 1) Silence
    if ((speech_prob < threshold) && (triggered == false)) {
      // printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
    }
    // 2) Speaking
    if ((speech_prob >= (threshold - 0.15)) && (triggered == true)) {
      // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample / sample_rate);
    }

    // 3) Start
    if ((speech_prob >= threshold) && (triggered == false)) {
      triggered = true;
      // minus window_size_samples to get precise start time point.
      current_speech.first = current_sample - window_size_samples;
    }

    // 4) End
    if ((speech_prob < (threshold - 0.15)) && (triggered == true)) {
      if (temp_end == 0) {
        temp_end = current_sample;
      }
      // a. silence < min_slience_samples, continue speaking
      if ((current_sample - temp_end) < min_silence_samples) {
        // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample /
        // sample_rate); printf("");
      }
      // b. silence >= min_slience_samples, end speaking
      else {
        current_speech.second = temp_end ? temp_end : current_sample;
        if (current_speech.second - current_speech.first > min_speech_samples) {
          speeches.push_back(current_speech);
        }
        current_speech = std::make_pair(-1.0f, -1.0f);
        temp_end = 0;
        triggered = false;
      }
    }
  }

  if ((current_speech.first >= 0.0f && current_speech.second < 0.0f) &&
      (input_wav.size() - current_speech.first > min_speech_samples)) {
    current_speech.second = input_wav.size();
    speeches.push_back(current_speech);
  }

  // add speech_pad_samples on both side of each speeches
  // carefully handle the boundary, without OOM and without overlapping
  for (int i = 0; i < speeches.size(); ++i) {
    if (i == 0) {
      speeches[i].first =
          std::max(0.0f, speeches[i].first - speech_pad_samples);
    }
    if (i != speeches.size() - 1) {
      int silence_duration = speeches[i + 1].first - speeches[i].second;
      if (silence_duration < 2 * speech_pad_samples) {
        speeches[i].second += silence_duration / 2;
        speeches[i + 1].first =
            std::max(0.0f, speeches[i + 1].first - silence_duration / 2);
      } else {
        speeches[i].second = std::min(static_cast<float>(input_wav.size()),
                                      speeches[i].second + speech_pad_samples);
        speeches[i + 1].first =
            std::max(0.0f, speeches[i + 1].first - speech_pad_samples);
      }
    } else {
      speeches[i].second = std::min(static_cast<float>(input_wav.size()),
                                    speeches[i].second + speech_pad_samples);
    }
  }

  for (auto &speech : speeches) {
    if (return_seconds) {
      printf("{ start: %.3f s }\n", 1.0 * speech.first / sample_rate);
      printf("{ end: %.3f s }\n", 1.0 * speech.second / sample_rate);
    } else {
      printf("{ start: %.3f }\n", speech.first);
      printf("{ end: %.3f }\n", speech.second);
    }
  }

  // save segment wavs
  //

  std::filesystem::path output_path(output_dir);

  if (output_dir.empty()) {
    // 使用 wav_path 所在的目录作为输出目录
    std::filesystem::path path(wav_path);
    output_path = path.parent_path();
  }

  // Create the output directory if it does not exist
  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directories(output_path);
  }

  // Ensure the output directory ends with a slash
  if (!output_path.empty() && output_path.string().back() != '/') {
    output_path += '/';
  }

  size_t suffix_pos = wav_path.find_last_of("/\\");
  std::string base_name = (suffix_pos != std::string::npos)
                              ? wav_path.substr(suffix_pos + 1)
                              : wav_path;
  base_name = base_name.substr(0, base_name.rfind(".wav"));

  for (int i = 0; i < speeches.size(); ++i) {
    std::vector<float> segment_wav(input_wav.begin() + speeches[i].first,
                                   input_wav.begin() + speeches[i].second);

    for (size_t j = 0; j < segment_wav.size(); ++j) {
      segment_wav[j] *= 32768.0f;
    }

    std::string segment_wav_path =
        output_path.string() + base_name + "_" + std::to_string(i) + ".wav";

    wav::WavWriter wav_writer(segment_wav.data(), segment_wav.size(), 1,
                              sample_rate, 16);
    wav_writer.Write(segment_wav_path);
  }

  return;
}
