#!/usr/bin/env python3
"""Command-line interface for batch image ingestion.

Usage examples:
    # Process with Ollama (default)
    python ingest_cli.py /path/to/images

    # Process with OpenAI
    python ingest_cli.py /path/to/images --provider openai --api-key sk-...

    # Reprocess everything
    python ingest_cli.py /path/to/images --reprocess

    # Use custom Ollama URL and model
    python ingest_cli.py /path/to/images --ollama-url http://server:11434 --model llama3.2-vision:90b

    # Dry run (scan only, don't process)
    python ingest_cli.py /path/to/images --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Import the refactored ingestion module
import image_ingestion as ing


def print_banner(text: str) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def scan_directory(directory: Path) -> None:
    """Scan and report on directory contents without processing."""
    if not directory.exists() or not directory.is_dir():
        print(f"❌ Error: Directory does not exist: {directory}")
        sys.exit(1)
    
    images = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in ing.IMAGE_EXTENSIONS
    ]
    
    print_banner(f"SCAN RESULTS: {directory}")
    print(f"  Total images found: {len(images)}")
    
    if not images:
        print("  ⚠️  No images found")
        return
    
    # Count by extension
    from collections import Counter
    extensions = Counter(p.suffix.lower() for p in images)
    print("\n  Breakdown by extension:")
    for ext, count in extensions.most_common():
        print(f"    {ext}: {count} files")
    
    # Check for existing OCR/JSON files
    ocr_exists = sum(1 for img in images if ing._ocr_path_for_image(img).exists())
    json_exists = sum(1 for img in images if ing._json_path_for_image(img).exists())
    
    print(f"\n  Existing intermediate files:")
    print(f"    .ocr.txt files: {ocr_exists}")
    print(f"    .json files: {json_exists}")
    
    print("\n  Sample files:")
    for img in images[:5]:
        print(f"    {img.relative_to(directory)}")
    if len(images) > 5:
        print(f"    ... and {len(images) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Batch image ingestion for document OCR and structuring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing images to process"
    )
    
    # Provider selection
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="AI provider (default: from HISTORIAN_AGENT_MODEL_PROVIDER env or 'ollama')"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider-specific defaults)"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama base URL (default: from env or http://localhost:11434)"
    )
    
    # OpenAI configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or use OPENAI_API_KEY env variable)"
    )
    
    parser.add_argument(
        "--api-key-file",
        type=str,
        default=None,
        help="Path to file containing OpenAI API key"
    )
    
    # Processing options
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess all images, even if OCR/JSON files exist"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan directory and show what would be processed, but don't process"
    )
    
    # Prompt customization
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for structuring (default: built-in prompt)"
    )
    
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to file containing custom prompt"
    )
    
    # Output options
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Write summary JSON to this file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Expand directory path
    directory = ing.expand_directory(args.directory)
    
    # Handle dry run
    if args.dry_run:
        scan_directory(directory)
        return
    
    # Determine provider
    provider = args.provider or ing.provider_from_string(None)
    
    # Get model
    if args.model:
        model = args.model
    elif provider == "ollama":
        model = ing.DEFAULT_OLLAMA_MODEL
    else:
        model = ing.DEFAULT_OPENAI_MODEL
    
    # Get prompt
    prompt = ing.DEFAULT_PROMPT
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser()
        if not prompt_path.exists():
            print(f"❌ Error: Prompt file not found: {prompt_path}")
            sys.exit(1)
        prompt = prompt_path.read_text(encoding="utf-8")
    
    # Handle API key for OpenAI
    api_key: Optional[str] = None
    if provider == "openai":
        if args.api_key:
            api_key = args.api_key
        elif args.api_key_file:
            key_path = Path(args.api_key_file).expanduser()
            if not key_path.exists():
                print(f"❌ Error: API key file not found: {key_path}")
                sys.exit(1)
            api_key = key_path.read_text(encoding="utf-8").strip()
        else:
            api_key = ing.ensure_api_key(None)
        
        if not api_key:
            print("❌ Error: OpenAI API key required")
            print("   Provide via --api-key, --api-key-file, or OPENAI_API_KEY env")
            sys.exit(1)
    
    # Create config
    config = ing.ModelConfig(
        provider=provider,
        model=model,
        prompt=prompt,
        base_url=args.ollama_url,
        temperature=0.0,
        max_tokens=4000
    )
    
    # Print configuration
    print_banner("CONFIGURATION")
    print(f"  Directory: {directory}")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    if provider == "ollama":
        print(f"  Ollama URL: {args.ollama_url or ing.DEFAULT_OLLAMA_BASE_URL}")
        print(f"  Stage 2 Model: {ing.DEFAULT_OLLAMA_STAGE2_MODEL}")
    print(f"  Reprocess existing: {args.reprocess}")
    
    # Run ingestion
    try:
        summary = ing.process_directory(
            directory=directory,
            config=config,
            reprocess_existing=args.reprocess,
            api_key=api_key
        )
        
        # Print final summary
        print_banner("FINAL SUMMARY")
        print(f"  Total images: {summary.images_total}")
        print(f"  ✅ Generated: {summary.generated}")
        print(f"  📥 Queued: {summary.queued_existing}")
        print(f"  ⏭️  Skipped: {summary.skipped_existing}")
        print(f"  💾 Ingested: {summary.ingested}")
        print(f"  🔄 Updated: {summary.updated}")
        print(f"  ❌ Failed: {summary.failed}")
        
        if summary.errors:
            print(f"\n  Errors ({len(summary.errors)}):")
            for error in summary.errors[:10]:  # Show first 10
                print(f"    • {Path(error['path']).name}: {error['error'][:100]}")
            if len(summary.errors) > 10:
                print(f"    ... and {len(summary.errors) - 10} more errors")
        
        # Write summary to file if requested
        if args.summary_file:
            summary_path = Path(args.summary_file).expanduser()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"\n  📄 Summary written to: {summary_path}")
        
        # Exit with error code if there were failures
        if summary.failed > 0 or summary.ingest_failures > 0:
            sys.exit(1)
        
    except ing.IngestionError as exc:
        print(f"\n❌ Ingestion error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
