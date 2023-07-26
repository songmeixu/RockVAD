#------------------------- linux -------------------------
# onnx
bin/rockvad-onnx \
  --onnx-model=/audio/code/RockVAD/models/silero_vad.opt.onnx \
  /audio/data/roborock/nihao_shitou/wav/smx/46.wav

#------------------------- linux -------------------------
LD_LIBRARY_PATH=lib:LD_LIBRARY_PATH \
bin_release/rockvad-onnx-time \
  --onnx-model=models/silero_vad.opt.onnx \
  /mnt/data/meixu/work/npu/sherpa/wav/46.wav
