import argparse
import json
import os
import time

import tensorflow as tf

from config import ProjectConfig
from model_factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick architecture comparison (build + forward pass).")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["vgg16", "resnet50", "efficientnetb0"],
        help="Architectures to compare.",
    )
    parser.add_argument("--out", default="model_compare.json", help="Output JSON file.")
    args = parser.parse_args()

    cfg = ProjectConfig()
    input_shape = (cfg.image_size[0], cfg.image_size[1], 3)
    sample = tf.random.uniform((1, cfg.image_size[0], cfg.image_size[1], 3))

    results = []
    for arch in args.architectures:
        t0 = time.time()
        model = build_model(arch, input_shape=input_shape, num_classes=2)
        build_time_s = time.time() - t0

        t1 = time.time()
        _ = model(sample, training=False)
        infer_time_ms = (time.time() - t1) * 1000

        params = model.count_params()
        result = {
            "architecture": arch,
            "params": int(params),
            "build_time_s": round(build_time_s, 4),
            "single_forward_ms": round(infer_time_ms, 4),
        }
        results.append(result)
        print(result)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison to {args.out}")


if __name__ == "__main__":
    main()
