#!/usr/bin/env python
# 2025-11-25 14:30 EST
# Purpose: Batch OCR with Qwen/Qwen3-VL-32B-Instruct across multiple GPUs,
#          saving one .txt file per image and writing a detailed log file.

import os
import time
import traceback
import multiprocessing as mp

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================
#                   CONFIGURATION BLOCK
# ============================================================

# Model and decoding settings
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
TEMPERATURE = 0.0
DO_SAMPLE = False
MAX_NEW_TOKENS = 2048

# OCR system instruction
OCR_PROMPT = (
    "You are an OCR engine. Read every piece of text in this document image. "
    "Return ONLY the raw text content, preserving all line breaks. "
    "Do not describe the image or add extra words."
)

# Root directory to process (recursively)
INPUT_DIR = "/data/lhyman6/nosql_project/nosql/archives/"

# File extensions to treat as images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Multi-GPU / multi-process settings
MAX_GPUS_TO_USE = 4          # use up to this many GPUs if available
PROCESSES_PER_GPU = 1        # 1 process per GPU for memory stability

# Logging
PRINT_EVERY_N_IMAGES = 10    # print progress every N images
LOG_FILE = os.path.join(INPUT_DIR, "qwen_ocr_log.txt")  # single log file for this run


# ============================================================
#                     OCR HELPER FUNCTION
# ============================================================

def ocr_single_image(image_path: str, model, processor) -> str:
    """
    Perform OCR on a single image using an already-loaded model and processor.
    Returns the raw decoded text.
    """
    abs_path = os.path.abspath(image_path)

    # Build HF-style chat message with image + prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": abs_path},   # plain local path
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]

    # Tokenize and build inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

    # Strip prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode
    text_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return text_output


# ============================================================
#                     WORKER PROCESS LOOP
# ============================================================

def worker_loop(worker_id: int,
                gpu_slot: int,
                task_queue: mp.Queue,
                success_list,
                failure_list):
    """
    Worker process:
      - Loads model + processor once
      - Repeatedly pulls image paths from task_queue
      - Runs OCR and writes .txt files
      - Records successes/failures in shared lists
    """
    start_worker = time.time()
    
    print(f"[Worker {worker_id}] Starting on GPU {gpu_slot}")

    # Load model and processor
    try:
        t0 = time.time()
        print(f"[Worker {worker_id}] Loading model {MODEL_NAME} on GPU {gpu_slot}..")
        
        # Use device_map="sequential" to load model in chunks
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="sequential",  # Load model in chunks
        )
        # Explicitly move to the correct GPU
        model = model.to(f"cuda:{gpu_slot}")
        t1 = time.time()
        print(f"[Worker {worker_id}] Model loaded in {t1 - t0:.1f} s on GPU {gpu_slot}")

        print(f"[Worker {worker_id}] Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        t2 = time.time()
        print(f"[Worker {worker_id}] Processor loaded in {t2 - t1:.1f} s")
    except Exception as e:
        print(f"[Worker {worker_id}] FATAL: failed to load model/processor: {e}")
        traceback.print_exc()
        return

    # Main processing loop
    images_processed = 0
    while True:
        image_path = task_queue.get()
        if image_path is None:
            # Sentinel value indicates no more work
            print(f"[Worker {worker_id}] Received stop signal, exiting.")
            break

        images_processed += 1
        img_start = time.time()
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Skip if output already exists
            base, _ = os.path.splitext(os.path.abspath(image_path))
            output_path = base + ".txt"
            if os.path.exists(output_path):
                if images_processed % PRINT_EVERY_N_IMAGES == 0:
                    print(f"[Worker {worker_id}] Skipping {image_path} (output exists)")
                continue

            if images_processed % PRINT_EVERY_N_IMAGES == 0:
                print(f"[Worker {worker_id}] Starting OCR: {image_path}")

            text_output = ocr_single_image(image_path, model, processor)

            # Write output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_output)

            img_end = time.time()
            elapsed = img_end - img_start
            success_list.append((image_path, output_path, elapsed))

            if images_processed % PRINT_EVERY_N_IMAGES == 0:
                print(f"[Worker {worker_id}] DONE {image_path} → {output_path} "
                      f"in {elapsed:.2f} s")
        except Exception as e:
            img_end = time.time()
            elapsed = img_end - img_start
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[Worker {worker_id}] ERROR on {image_path}: {err_msg} "
                  f"(elapsed {elapsed:.2f} s)")
            traceback.print_exc()
            failure_list.append((image_path, err_msg, elapsed))

    total_time = time.time() - start_worker
    print(f"[Worker {worker_id}] Finished. Images processed: {images_processed}. "
          f"Total worker time: {total_time:.1f} s")


# ============================================================
#                     IMAGE DISCOVERY
# ============================================================

def find_images(root_dir: str):
    """
    Recursively collect all image files under root_dir
    matching IMAGE_EXTENSIONS.
    """
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, filename))
    return sorted(image_paths)


# ============================================================
#                     LOG WRITING
# ============================================================

def write_log(log_path: str,
              images,
              successes,
              failures,
              total_elapsed: float,
              num_gpus_to_use: int,
              num_workers: int):
    """
    Write a human-readable log file summarizing the run.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Qwen3-VL Batch OCR Log\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"Input directory: {INPUT_DIR}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"GPUs used: {num_gpus_to_use}\n")
        f.write(f"Workers: {num_workers}\n")
        f.write(f"Total images discovered: {len(images)}\n")
        f.write(f"Successful OCR: {len(successes)}\n")
        f.write(f"Failed OCR: {len(failures)}\n")
        f.write(f"Total elapsed wall time: {total_elapsed:.2f} s\n\n")

        f.write("=== Successes ===\n")
        if not successes:
            f.write("None\n")
        else:
            for img_path, out_path, elapsed in successes:
                f.write(f"SUCCESS | {elapsed:.2f} s | {img_path} -> {out_path}\n")

        f.write("\n=== Failures ===\n")
        if not failures:
            f.write("None\n")
        else:
            for img_path, err_msg, elapsed in failures:
                f.write(f"FAIL    | {elapsed:.2f} s | {img_path} | {err_msg}\n")


# ============================================================
#                     MAIN EXECUTION
# ============================================================

def main():
    print("=== Qwen3-VL Batch OCR Starting ===")
    print(f"Input directory (recursive): {INPUT_DIR}")
    print(f"Log file: {LOG_FILE}")

    # Discover images
    t0 = time.time()
    images = find_images(INPUT_DIR)
    t1 = time.time()
    print(f"Found {len(images)} image(s) in {t1 - t0:.2f} s")

    if not images:
        print("No images found. Exiting.")
        return

    # Determine how many GPUs we can use
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No CUDA GPUs detected. Exiting.")
        return

    num_gpus_to_use = min(MAX_GPUS_TO_USE, available_gpus)
    num_workers = num_gpus_to_use * PROCESSES_PER_GPU

    print(f"CUDA devices visible: {available_gpus}")
    print(f"Using up to {num_gpus_to_use} GPU(s)")
    print(f"Spawning {num_workers} worker process(es) "
          f"({PROCESSES_PER_GPU} per GPU)")

    # Shared structures for communication
    manager = mp.Manager()
    task_queue = manager.Queue()
    success_list = manager.list()
    failure_list = manager.list()

    # Enqueue all image paths
    for img in images:
        task_queue.put(img)

    # Add sentinel values (one per worker) to signal completion
    for _ in range(num_workers):
        task_queue.put(None)

    # Spawn workers
    workers = []
    for worker_id in range(num_workers):
        gpu_slot = worker_id % num_gpus_to_use
        p = mp.Process(
            target=worker_loop,
            args=(worker_id, gpu_slot, task_queue, success_list, failure_list),
        )
        p.start()
        workers.append(p)

    # Wait for all workers to finish
    for p in workers:
        p.join()

    t_end = time.time()
    total_elapsed = t_end - t0

    # Summarize results
    successes = list(success_list)
    failures = list(failure_list)

    print("\n=== Qwen3-VL Batch OCR Summary ===")
    print(f"Total images discovered: {len(images)}")
    print(f"Successful OCR: {len(successes)}")
    print(f"Failed OCR: {len(failures)}")
    print(f"Total elapsed wall time: {total_elapsed:.2f} s")

    if failures:
        print("\nFailures:")
        for img_path, err_msg, elapsed in failures:
            print(f"  - {img_path} ({elapsed:.2f} s): {err_msg}")

    # Write log file
    print(f"\nWriting log to: {LOG_FILE}")
    write_log(
        log_path=LOG_FILE,
        images=images,
        successes=successes,
        failures=failures,
        total_elapsed=total_elapsed,
        num_gpus_to_use=num_gpus_to_use,
        num_workers=num_workers,
    )
    print("Log written.")
    print("===================================\n")


if __name__ == "__main__":
    main()
