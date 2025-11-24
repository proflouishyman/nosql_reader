#!/usr/bin/env python3
"""Parallel batch processing for multiple directories.

This script processes multiple directories in parallel on a single machine
using Python's multiprocessing. Useful for workstations or testing before
running on HPC.

Usage:
    # Process all subdirectories in parallel (4 workers)
    python ingest_parallel.py /path/to/images --workers 4

    # Process specific directories
    python ingest_parallel.py /path/to/dir1 /path/to/dir2 /path/to/dir3 --workers 2

    # Dry run to see what would be processed
    python ingest_parallel.py /path/to/images --dry-run
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional

import image_ingestion as ing


def find_subdirectories(root: Path, max_depth: int = 1) -> List[Path]:
    """Find all subdirectories up to max_depth."""
    subdirs = []
    for path in root.iterdir():
        if path.is_dir():
            subdirs.append(path)
            if max_depth > 1:
                subdirs.extend(find_subdirectories(path, max_depth - 1))
    return sorted(subdirs)


def process_single_directory(args_tuple):
    """Process a single directory (for multiprocessing)."""
    directory, config, reprocess, api_key, worker_id = args_tuple
    
    print(f"\n[Worker {worker_id}] Starting: {directory.name}")
    
    try:
        summary = ing.process_directory(
            directory=directory,
            config=config,
            reprocess_existing=reprocess,
            api_key=api_key
        )
        
        print(f"\n[Worker {worker_id}] Completed: {directory.name}")
        print(f"  Generated: {summary.generated}, Ingested: {summary.ingested}, Failed: {summary.failed}")
        
        return {
            "directory": str(directory),
            "success": True,
            "summary": summary.as_dict()
        }
        
    except Exception as exc:
        print(f"\n[Worker {worker_id}] ❌ Failed: {directory.name}")
        print(f"  Error: {exc}")
        
        return {
            "directory": str(directory),
            "success": False,
            "error": str(exc)
        }


def scan_and_report(directories: List[Path]) -> None:
    """Scan directories and report what would be processed."""
    print("\n" + "=" * 70)
    print("  DRY RUN - Scanning directories")
    print("=" * 70)
    
    total_images = 0
    
    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            print(f"\n❌ Not a directory: {directory}")
            continue
        
        images = [
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in ing.IMAGE_EXTENSIONS
        ]
        
        total_images += len(images)
        
        print(f"\n📁 {directory.name}")
        print(f"   Images: {len(images)}")
        
        if images:
            # Check existing files
            ocr_exists = sum(1 for img in images if ing._ocr_path_for_image(img).exists())
            json_exists = sum(1 for img in images if ing._json_path_for_image(img).exists())
            print(f"   Existing .ocr.txt: {ocr_exists}")
            print(f"   Existing .json: {json_exists}")
    
    print("\n" + "=" * 70)
    print(f"  Total: {len(directories)} directories, {total_images} images")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Process multiple directories in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="Directories to process (or single parent directory)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--subdirs",
        action="store_true",
        help="If single directory given, process all subdirectories in parallel"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Max depth for subdirectory scanning (default: 1)"
    )
    
    # Provider and model options
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="AI provider"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama base URL"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key"
    )
    
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess existing files"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without processing"
    )
    
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Write combined summary to this file"
    )
    
    args = parser.parse_args()
    
    # Determine directories to process
    directories: List[Path] = []
    
    if len(args.directories) == 1 and args.subdirs:
        # Process subdirectories of single given directory
        root = ing.expand_directory(args.directories[0])
        if not root.exists() or not root.is_dir():
            print(f"❌ Error: Not a directory: {root}")
            sys.exit(1)
        directories = find_subdirectories(root, args.max_depth)
        print(f"Found {len(directories)} subdirectories in {root}")
    else:
        # Process explicitly listed directories
        directories = [ing.expand_directory(d) for d in args.directories]
    
    if not directories:
        print("❌ Error: No directories to process")
        sys.exit(1)
    
    # Dry run
    if args.dry_run:
        scan_and_report(directories)
        return
    
    # Setup configuration
    provider = args.provider or ing.provider_from_string(None)
    
    if args.model:
        model = args.model
    elif provider == "ollama":
        model = ing.DEFAULT_OLLAMA_MODEL
    else:
        model = ing.DEFAULT_OPENAI_MODEL
    
    api_key: Optional[str] = None
    if provider == "openai":
        api_key = args.api_key or ing.ensure_api_key(None)
        if not api_key:
            print("❌ Error: OpenAI API key required")
            sys.exit(1)
    
    config = ing.ModelConfig(
        provider=provider,
        model=model,
        prompt=ing.DEFAULT_PROMPT,
        base_url=args.ollama_url,
        temperature=0.0,
        max_tokens=4000
    )
    
    # Print configuration
    print("\n" + "=" * 70)
    print("  PARALLEL BATCH PROCESSING")
    print("=" * 70)
    print(f"  Directories: {len(directories)}")
    print(f"  Workers: {args.workers}")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    if provider == "ollama":
        print(f"  Stage 2 Model: {ing.DEFAULT_OLLAMA_STAGE2_MODEL}")
    print(f"  Reprocess: {args.reprocess}")
    print("=" * 70)
    
    # Prepare arguments for each worker
    work_items = [
        (directory, config, args.reprocess, api_key, i)
        for i, directory in enumerate(directories, 1)
    ]
    
    # Process in parallel
    start_time = time.time()
    
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(process_single_directory, work_items)
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes
    
    total_images = 0
    total_generated = 0
    total_ingested = 0
    total_failed = 0
    
    for result in results:
        if result["success"]:
            summary = result["summary"]
            total_images += summary["images_total"]
            total_generated += summary["generated"]
            total_ingested += summary["ingested"]
            total_failed += summary["failed"]
    
    # Print final summary
    print("\n" + "=" * 70)
    print("  BATCH COMPLETE")
    print("=" * 70)
    print(f"  Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Directories processed: {successes}/{len(directories)}")
    print(f"  Total images: {total_images}")
    print(f"  Generated: {total_generated}")
    print(f"  Ingested: {total_ingested}")
    print(f"  Failed: {total_failed}")
    
    if failures > 0:
        print(f"\n  ❌ Failed directories: {failures}")
        for result in results:
            if not result["success"]:
                print(f"    • {Path(result['directory']).name}: {result.get('error', 'Unknown error')}")
    
    # Save summary
    if args.summary_file:
        summary_path = Path(args.summary_file).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({
                "elapsed_seconds": elapsed,
                "directories_processed": successes,
                "directories_failed": failures,
                "total_images": total_images,
                "total_generated": total_generated,
                "total_ingested": total_ingested,
                "total_failed": total_failed,
                "results": results
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n  📄 Summary written to: {summary_path}")
    
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
