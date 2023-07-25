# issues

- about onnxruntime warning: **CleanUnusedInitializersAndNodeArgs**

  ```bash
  # warn
  2023-07-25 15:16:25.003829668 [W:onnxruntime:, graph.cc:3543 CleanUnusedInitializersAndNodeArgs] Removing initializer '617'. It is not used by any node and should be removed from the model.
  
  # solution: use onnxruntime load and save optimized model
  conda activate silero
  python /audio/code/onnxruntime/onnxruntime/python/tools/transformers/optimizer.py \
   --input /audio/code/RockVAD/models/silero_vad.onnx \
   --output /audio/code/RockVAD/models/silero_vad.opt.onnx \
   --opt_level 1 \
   --only_onnxruntime
  ```

  - 注: 不要再使用onnxsim, onnxoptimizer等工具, 会导致继续warn/error (20230725)

---
