import argparse
import os
import time

import torch

from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingPipeline,
    SkyReelsV2Transformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    quantize_,
)


MODEL_ID = "Skywork/SkyReels-V2-DF-14B-540P-Diffusers"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    print("SkyReels model:", MODEL_ID)
    print("GPU:", torch.cuda.get_device_name(0))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.time()

    print("[1/6] Loading transformer...")

    transformer = SkyReelsV2Transformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print("[2/6] Quantizing transformer FP8...")

    transformer = transformer.to("cuda")

    quantize_(
        transformer,
        Float8DynamicActivationFloat8WeightConfig(),
    )

    print("[3/6] Loading VAE + pipeline...")

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=8.0,
    )

    print("[4/6] Enabling aggressive group offload...")

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    apply_group_offloading(
        pipe.text_encoder,
        onload_device=cuda,
        offload_device=cpu,
        offload_type="block_level",
        num_blocks_per_group=4,
    )

    pipe.transformer.enable_group_offload(
        onload_device=cuda,
        offload_device=cpu,
        offload_type="leaf_level",
        use_stream=True,
    )

    pipe.vae.enable_group_offload(
        onload_device=cuda,
        offload_device=cpu,
        offload_type="leaf_level",
    )

    pipe.vae.enable_tiling()

    torch.cuda.empty_cache()

    print("[5/6] Generating 57 frames / 544x960...")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        height=544,
        width=960,
        num_frames=57,
        base_num_frames=57,
        num_inference_steps=20,
        ar_step=5,
        causal_block_size=5,
        overlap_history=None,
        addnoise_condition=20,
        generator=generator,
    )

    print("[6/6] Encoding MP4...")

    export_to_video(
        result.frames[0],
        args.output,
        fps=24,
        quality=8,
    )

    print("Generation seconds:", round(time.time() - started, 2))
    print(
        "Peak VRAM GB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    )
    print("SKYREELS READY ✅")


if __name__ == "__main__":
    main()
