#!/usr/bin/env python
# 2025-11-25 10:45 EST
# Purpose: OCR script using Qwen/Qwen3-VL-32B-Instruct,
#          saving output as a .txt file next to the image.
#works

import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================
#                   CONFIGURATION BLOCK
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

# Deterministic behavior — no creative sampling.
TEMPERATURE = 0.0
DO_SAMPLE = False

# Large window to handle dense documents.
MAX_NEW_TOKENS = 2048

# OCR system instruction.
OCR_PROMPT = (
    "You are an OCR engine. Read every piece of text in this document image. "
    "Return ONLY the raw text content, preserving all line breaks. "
    "Do not describe the image or add extra words."
)

# Hardcode your image path here OR call `ocr_to_file(path)`
IMAGE_PATH = "/data/lhyman6/nosql_project/nosql/archives/Paper_mini/RDApp-592563Morton004.jpg"


# ============================================================
#                   CORE OCR FUNCTION
# ============================================================

def ocr_to_file(image_path: str) -> str:
    """
    Performs OCR on `image_path` using Qwen3-VL-32B-Instruct and
    writes a .txt file with the same base name next to the image.

    Returns the path to the written output file.
    """

    print("\n=== OCR START ===")

    # ---- 1. Prepare image path (NO file://) ----
    print(f"[1] Resolving image path: {image_path}")
    abs_path = os.path.abspath(image_path)
    print(f"     Absolute path: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image file does not exist: {abs_path}")

    # IMPORTANT CHANGE: pass plain path, not file:// URI
    image_source = abs_path
    print(f"     Image source passed to model: {image_source}")

    # ---- 2. Compute output text file path ----
    base, _ = os.path.splitext(abs_path)
    output_path = base + ".txt"
    print(f"[2] Output will be written to: {output_path}")

    # ---- 3. Load model + processor ----
    print(f"[3] Loading model: {MODEL_NAME}")
    print("     This may take several minutes the first time...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
    )
    print("     Model loaded.")

    print("[3b] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("     Processor loaded.")

    # ---- 4. Build HF message format ----
    print("[4] Building messages for inference...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source},  # <-- plain path
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]
    print("     Messages built.")

    # ---- 5. Tokenize & build model inputs ----
    print("[5] Applying chat template and tokenizing...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    print("     Tokenization complete.")

    print("[5b] Moving tensors to device(s)...")
    inputs = inputs.to(model.device)
    print(f"     Using device: {model.device}")

    # ---- 6. Run inference ----
    print("[6] Generating OCR text (may take time)...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )
    print("     Generation complete.")

    # ---- 7. Trim input prompt tokens from output ----
    print("[7] Trimming prompt tokens...")
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print("     Trim complete.")

    # ---- 8. Decode text ----
    print("[8] Decoding generated text...")
    text_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("     Decode complete.")

    # ---- 9. Write result to output file ----
    print("[9] Writing result to output file...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_output)
    print("     File written.")

    print("=== OCR COMPLETE ===\n")
    return output_path


# ============================================================
#                     MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Running OCR pipeline...")
    saved = ocr_to_file(IMAGE_PATH)
    print(f"Output saved to: {saved}")
