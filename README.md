# RockVAD

```bash
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
```
