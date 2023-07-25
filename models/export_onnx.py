#!/usr/bin/env python3
#
# Copyright 2023 Roborock (Author: Meixu Song)


import logging
import torch


def main():
    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    torch.set_grad_enabled(False)
    jit_model = torch.jit.load("silero_vad.pt", map_location=device)
    jit_model.eval()

    # export the model to ONNX
    logging.info("Using torch.onnx.export")
    filename = "silero_vad.b1.onnx"

    window_size = 1536
    x = torch.rand(1, window_size, dtype=torch.float32).to(device)
    x = 2 * x - 1

    sampling_rate = 16000
    sr = torch.tensor([sampling_rate], dtype=torch.int64).to(device)

    h = torch.zeros(2, 1, 64, dtype=torch.float32).to(device)
    c = torch.zeros(2, 1, 64, dtype=torch.float32).to(device)

    input_names = ["input", "sr", "h", "c"]
    output_names = ["output", "hn", "cn"]

    # inputs = {}
    # outputs = {}
    # logging.info(f"h.shape: {h.shape}")
    # inputs["h"] = {1: "N"}
    # inputs["c"] = {1: "N"}
    # outputs["hn"] = {1: "N"}
    # outputs["cn"] = {1: "N"}

    # speech_prob = jit_model(x, sampling_rate).item()
    # print(f"speech_prob: {speech_prob}")

    torch.onnx.export(
        jit_model,
        (x, sr, h, c),
        # (x, {"sr": sr, "h": h, "c": c}),
        filename,
        verbose=True,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
    )

    logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
