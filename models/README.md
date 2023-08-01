# issues

- firstly, opt onnx model with onnxsim:

  ```bash
  onnxsim silero_vad.onnx silero_vad.onnxsim.onnx
  ```

- about onnxruntime warning: **CleanUnusedInitializersAndNodeArgs**

  ```bash
  # warn
  2023-07-25 15:16:25.003829668 [W:onnxruntime:, graph.cc:3543 CleanUnusedInitializersAndNodeArgs] Removing initializer '617'. It is not used by any node and should be removed from the model.

  # solution: use onnxruntime load and save optimized model
  conda activate silero
  python /audio/code/onnxruntime/onnxruntime/python/tools/transformers/optimizer.py \
   --input /audio/code/RockVAD/models/silero_vad.onnxsim.onnx \
   --output /audio/code/RockVAD/models/silero_vad.onnxsim.opt.onnx \
   --opt_level 1 \
   --only_onnxruntime
  ```

  - 注: 不要使用onnxoptimizer工具, 会导致error (20230725)

---

- convert onnx to ort

  ```bash
  conda activate onnx
  python -m onnxruntime.tools.convert_onnx_models_to_ort silero_vad.onnxsim.opt.onnx --enable_type_reduction --target_platform arm
  ```
