#------------------------- linux -------------------------
# onnx
bin/rockvad-onnx \
  --onnx-model=/audio/code/RockVAD/models/silero_vad.onnxsim.opt.onnx \
  /audio/data/roborock/nihao_shitou/wav/smx/46.wav

# ort
bin/rockvad-onnx \
  --onnx-model=/audio/code/RockVAD/models/silero_vad.onnxsim.opt.ort \
  /audio/data/roborock/nihao_shitou/wav/smx/46.wav

# segment wav
bin/rockvad-onnx \
  --segment-wav=true \
  --onnx-model=/audio/code/RockVAD/models/silero_vad.onnxsim.opt.ort \
  /audio/data/roborock/nihao_shitou/wav/smx/46.wav

bin/rockvad-onnx \
  --segment-wav=true \
  --threshold=0.3 \
  --onnx-model=/audio/code/RockVAD/models/silero_vad.onnxsim.opt.ort \
  /audio/work/denoise/data/语音录音文件上传速度测试/百3录音/唤醒测试集/9/changjing9_21.wav

bin/rockvad-onnx \
  --segment-wav=true \
  --threshold=0.5 \
  --onnx-model=/audio/code/RockVAD/models/silero_vad.onnxsim.opt.ort \
  /audio/work/denoise/data/语音录音文件上传速度测试/百3录音/离线指令/7/changjing7_9.wav

#------------------------- ARM64 -------------------------
# onnx
LD_LIBRARY_PATH=lib:LD_LIBRARY_PATH \
bin_release/rockvad-onnx-time \
  --onnx-model=models/silero_vad.opt.onnx \
  /mnt/data/meixu/work/npu/sherpa/wav/46.wav

# ort
LD_LIBRARY_PATH=lib/Release:LD_LIBRARY_PATH \
bin_release/rockvad-onnx-time \
  --onnx-model=models/silero_vad.opt.ort \
  /mnt/data/meixu/work/npu/sherpa/wav/46.wav
