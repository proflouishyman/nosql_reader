#!/usr/bin/env python
# 2025-11-25 14:30 EST
# Purpose: Batch OCR with Qwen/Qwen3-VL-8B-Instruct across multiple GPUs,
#          saving one .txt file per image and writing a detailed log file.

import os
import time
import traceback
import multiprocessing as mp

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Optional NVML for precise GPU memory reporting
try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


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
PROCESSES_PER_GPU = 1        # used if AUTOSCALE_PROCESSES_PER_GPU = False

# Autoscaling: choose processes per GPU based on VRAM
AUTOSCALE_PROCESSES_PER_GPU = True
TARGET_MODEL_MEM_GB = 18.0          # approximate VRAM per Qwen-8B model instance
MAX_PROCESSES_PER_GPU_AUTOSCALE = 8
MIN_PROCESSES_PER_GPU_AUTOSCALE = 1

# Logging
PRINT_EVERY_N_IMAGES = 10    # print progress every N images
LOG_FILE = os.path.join(INPUT_DIR, "qwen_ocr_log.txt")  # single log file for this run


# ============================================================
#                 GPU MEMORY REPORTING HELPERS
# ============================================================

def init_pynvml_once():
    """Initialize NVML if available."""
    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
        except Exception:
            pass


def get_gpu_memory_info(gpu_index: int, allow_torch: bool = True):
    """
    Return (used_gb, total_gb, free_gb) for a given GPU index.

    Uses NVML if available; otherwise falls back to PyTorch estimates
    *only* if allow_torch is True. In worker processes we pass
    allow_torch=False to avoid CUDA re-init issues.
    """
    # First try NVML
    if _HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = info.used / 1024**3
            total = info.total / 1024**3
            free = info.free / 1024**3
            return used, total, free
        except Exception:
            # Fall through to torch or zeros
            pass

    # Optional PyTorch fallback (only in main process)
    if allow_torch and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(gpu_index)
        total = props.total_memory / 1024**3
        used = torch.cuda.memory_allocated(gpu_index) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_index) / 1024**3
        free = max(total - max(used, reserved), 0.0)
        return used, total, free

    # If no NVML and no torch allowed, return zeros
    return 0.0, 0.0, 0.0


def print_gpu_memory_report(prefix: str = "", allow_torch: bool = True):
    """
    Print memory usage for all visible GPUs.

    In main: allow_torch=True → NVML or torch.
    In workers: allow_torch=False → NVML only (no torch.cuda calls).
    """
    if not torch.cuda.is_available():
        print(f"{prefix}No CUDA GPUs available for memory report.")
        return

    device_count = torch.cuda.device_count()
    print(f"{prefix}GPU memory report (visible devices: {device_count}):")
    for i in range(device_count):
        used, total, free = get_gpu_memory_info(i, allow_torch=allow_torch)
        # If we got all zeros and torch not allowed and no NVML, just say "unknown"
        if total == 0.0 and used == 0.0 and free == 0.0:
            print(f"{prefix}  GPU {i}: memory usage unknown (no NVML, torch disabled)")
        else:
            print(
                f"{prefix}  GPU {i}: used={used:.2f} GB, free={free:.2f} GB, total={total:.2f} GB"
            )


def autoscale_processes_per_gpu(num_gpus_to_use: int) -> int:
    """
    Decide how many processes per GPU to use based on free VRAM.

    Uses TARGET_MODEL_MEM_GB and caps at MAX_PROCESSES_PER_GPU_AUTOSCALE.
    This runs in the main process, so allow_torch=True is safe.
    """
    if not torch.cuda.is_available():
        return MIN_PROCESSES_PER_GPU_AUTOSCALE

    per_gpu_capacities = []
    for gpu_idx in range(num_gpus_to_use):
        used, total, free = get_gpu_memory_info(gpu_idx, allow_torch=True)
        if TARGET_MODEL_MEM_GB <= 0:
            capacity = MAX_PROCESSES_PER_GPU_AUTOSCALE
        else:
            capacity = int(free // TARGET_MODEL_MEM_GB)
        capacity = max(capacity, MIN_PROCESSES_PER_GPU_AUTOSCALE)
        capacity = min(capacity, MAX_PROCESSES_PER_GPU_AUTOSCALE)
        per_gpu_capacities.append(capacity)
        print(
            f"[Autoscale] GPU {gpu_idx}: used={used:.2f} GB, free={free:.2f} GB, "
            f"total={total:.2f} GB → capacity estimate={capacity} processes"
        )

    chosen = min(per_gpu_capacities) if per_gpu_capacities else MIN_PROCESSES_PER_GPU_AUTOSCALE
    print(f"[Autoscale] Chosen PROCESSES_PER_GPU = {chosen}")
    return chosen


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
    # In workers, NO torch.cuda queries – NVML-only, or "unknown"
    print_gpu_memory_report(prefix=f"[Worker {worker_id}] BEFORE LOAD | ",
                            allow_torch=False)

    # Load model and processor
    try:
        t0 = time.time()
        print(f"[Worker {worker_id}] Loading model {MODEL_NAME} on GPU {gpu_slot}..")

        # Load in chunks, then move to the specific GPU
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,          # use dtype instead of torch_dtype (deprecation)
            device_map="sequential",      # load model shards in order
        )
        model = model.to(f"cuda:{gpu_slot}")

        t1 = time.time()
        print(f"[Worker {worker_id}] Model loaded in {t1 - t0:.1f} s on GPU {gpu_slot}")
        print_gpu_memory_report(prefix=f"[Worker {worker_id}] AFTER MODEL LOAD | ",
                                allow_torch=False)

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
                print_gpu_memory_report(prefix=f"[Worker {worker_id}] BEFORE OCR | ",
                                        allow_torch=False)

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
                print_gpu_memory_report(prefix=f"[Worker {worker_id}] AFTER OCR | ",
                                        allow_torch=False)
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
    print_gpu_memory_report(prefix=f"[Worker {worker_id}] FINAL MEM | ",
                            allow_torch=False)


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
              num_workers: int,
              processes_per_gpu: int):
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
        f.write(f"Processes per GPU: {processes_per_gpu}\n")
        f.write(f"Autoscale enabled: {AUTOSCALE_PROCESSES_PER_GPU}\n")
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

    init_pynvml_once()
    print_gpu_memory_report(prefix="[Main] INITIAL | ", allow_torch=True)

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

    if AUTOSCALE_PROCESSES_PER_GPU:
        processes_per_gpu = autoscale_processes_per_gpu(num_gpus_to_use)
    else:
        processes_per_gpu = PROCESSES_PER_GPU

    num_workers = num_gpus_to_use * processes_per_gpu

    print(f"CUDA devices visible: {available_gpus}")
    print(f"Using up to {num_gpus_to_use} GPU(s)")
    print(f"Processes per GPU: {processes_per_gpu}")
    print(f"Spawning {num_workers} worker process(es)")

    print_gpu_memory_report(prefix="[Main] BEFORE WORKERS | ", allow_torch=True)

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
    print_gpu_memory_report(prefix="[Main] FINAL | ", allow_torch=True)

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
        processes_per_gpu=processes_per_gpu,
    )
    print("Log written.")
    print("===================================\n")


if __name__ == "__main__":
    main()
