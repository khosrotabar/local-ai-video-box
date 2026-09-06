import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 150

ROLE_INSTRUCTIONS = {
    "reference": (
        "Describe visually important subjects, appearance, composition, "
        "environment, and style."
    ),
    "character": (
        "Describe only visible identity-preserving traits: face, hair, "
        "apparent age range, clothing, body appearance, accessories, and "
        "distinctive features. Do not invent unseen attributes."
    ),
    "object": (
        "Describe shape, proportions, color, materials, markings or logos, "
        "and distinctive visual properties."
    ),
    "style": (
        "Describe visual style only: lighting, color palette, contrast, "
        "lens or look, texture, rendering style, and cinematic qualities. "
        "Avoid unrelated scene content."
    ),
    "location": (
        "Describe the environment: architecture or terrain, weather, "
        "lighting, time of day, atmosphere, and important spatial "
        "characteristics."
    ),
    "start_image": (
        "Give a short scene description for consistency with this primary "
        "image. Preserve only visible details."
    ),
    "final_target": (
        "Describe the exact visible geometry, silhouette, proportions, "
        "colors, stroke or line structure, and spacing. Do not infer or "
        "name brands. Do not reinterpret a symbol as a generic letter. "
        "Describe only what is visibly present and state that this is the "
        "exact desired final appearance."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_requests(path: Path):
    try:
        requests = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid analysis input.") from exc

    if not isinstance(requests, list) or not requests:
        raise ValueError("Analysis input must contain references.")

    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Invalid reference request.")

        if request.get("role") not in ROLE_INSTRUCTIONS:
            raise ValueError("Invalid reference role.")

        if not isinstance(request.get("upload_id"), str):
            raise ValueError("Invalid upload ID.")

        if not isinstance(request.get("image_path"), str):
            raise ValueError("Invalid image path.")

    return requests


def analyze_image(model, processor, request: dict) -> dict:
    image_path = Path(request["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError("Reference image is unavailable.")

    role = request["role"]
    instruction = ROLE_INSTRUCTIONS[role]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": (
                        "Analyze this image for video-generation reference "
                        "guidance. "
                        f"{instruction} "
                        "Be concise, factual, and describe only visible "
                        "information. Return one compact paragraph without "
                        "headings."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]
    analysis = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "upload_id": request["upload_id"],
        "role": role,
        "analysis": analysis,
    }


def main():
    args = parse_args()
    requests = load_requests(Path(args.input))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reference analysis.")

    print("Loading local reference analyzer:", MODEL_ID, flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    results = []

    for index, request in enumerate(requests, start=1):
        print(f"Analyzing reference {index}/{len(requests)}...", flush=True)
        results.append(analyze_image(model, processor, request))

    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Reference analysis complete.", flush=True)


if __name__ == "__main__":
    main()
