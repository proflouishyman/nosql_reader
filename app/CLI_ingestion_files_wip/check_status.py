#!/usr/bin/env python3
"""Status checker for batch ingestion operations.

This script scans directories and reports on ingestion progress:
- How many images exist
- How many have .ocr.txt files
- How many have .json files
- Which files failed or are missing

Useful for monitoring long-running batch jobs and identifying issues.

Usage:
    # Check status of a directory
    python check_status.py /path/to/images

    # Check multiple directories
    python check_status.py /path/to/archive1 /path/to/archive2

    # Export detailed report
    python check_status.py /path/to/images --report status_report.json

    # Find incomplete files
    python check_status.py /path/to/images --show-incomplete
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import image_ingestion as ing


def analyze_directory(directory: Path) -> Dict:
    """Analyze processing status of a directory."""
    
    if not directory.exists() or not directory.is_dir():
        return {
            "directory": str(directory),
            "error": "Directory not found or not a directory",
            "exists": False
        }
    
    # Find all images
    images = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in ing.IMAGE_EXTENSIONS
    ]
    
    # Analyze each image
    complete = []
    has_ocr = []
    has_json = []
    needs_processing = []
    
    for img in images:
        ocr_path = ing._ocr_path_for_image(img)
        json_path = ing._json_path_for_image(img)
        
        ocr_exists = ocr_path.exists()
        json_exists = json_path.exists()
        
        if json_exists:
            complete.append(img)
        elif ocr_exists:
            has_ocr.append(img)
        else:
            needs_processing.append(img)
        
        if ocr_exists:
            has_ocr.append(img)
        if json_exists:
            has_json.append(img)
    
    # Extension breakdown
    extensions = Counter(p.suffix.lower() for p in images)
    
    return {
        "directory": str(directory),
        "exists": True,
        "total_images": len(images),
        "complete": len(complete),
        "has_ocr_only": len([p for p in images if ing._ocr_path_for_image(p).exists() and not ing._json_path_for_image(p).exists()]),
        "needs_processing": len(needs_processing),
        "extensions": dict(extensions),
        "complete_files": [str(p.relative_to(directory)) for p in complete],
        "incomplete_files": [str(p.relative_to(directory)) for p in needs_processing],
        "needs_structuring": [str(p.relative_to(directory)) for p in has_ocr if p not in complete],
        "completion_rate": (len(complete) / len(images) * 100) if images else 0.0
    }


def print_directory_status(analysis: Dict, show_details: bool = False) -> None:
    """Print status for a single directory."""
    
    if not analysis.get("exists"):
        print(f"\n❌ {analysis['directory']}")
        print(f"   Error: {analysis.get('error', 'Unknown error')}")
        return
    
    total = analysis["total_images"]
    complete = analysis["complete"]
    has_ocr = analysis["has_ocr_only"]
    needs = analysis["needs_processing"]
    rate = analysis["completion_rate"]
    
    # Status emoji
    if rate == 100:
        status = "✅"
    elif rate > 50:
        status = "🟡"
    elif rate > 0:
        status = "🟠"
    else:
        status = "⚪"
    
    print(f"\n{status} {Path(analysis['directory']).name}")
    print(f"   Total images: {total}")
    print(f"   Complete (.json): {complete} ({rate:.1f}%)")
    print(f"   OCR only (.ocr.txt): {has_ocr}")
    print(f"   Needs processing: {needs}")
    
    if analysis["extensions"]:
        print(f"   Extensions: {', '.join(f'{ext}({count})' for ext, count in analysis['extensions'].items())}")
    
    if show_details and analysis["incomplete_files"]:
        print(f"\n   Incomplete files:")
        for file in analysis["incomplete_files"][:10]:
            print(f"     • {file}")
        if len(analysis["incomplete_files"]) > 10:
            print(f"     ... and {len(analysis['incomplete_files']) - 10} more")
    
    if show_details and analysis["needs_structuring"]:
        print(f"\n   Has OCR, needs JSON structuring:")
        for file in analysis["needs_structuring"][:10]:
            print(f"     • {file}")
        if len(analysis["needs_structuring"]) > 10:
            print(f"     ... and {len(analysis['needs_structuring']) - 10} more")


def print_summary(analyses: List[Dict]) -> None:
    """Print overall summary."""
    
    total_dirs = len(analyses)
    valid_dirs = sum(1 for a in analyses if a.get("exists"))
    total_images = sum(a.get("total_images", 0) for a in analyses)
    total_complete = sum(a.get("complete", 0) for a in analyses)
    total_needs = sum(a.get("needs_processing", 0) for a in analyses)
    
    overall_rate = (total_complete / total_images * 100) if total_images > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Directories: {valid_dirs}/{total_dirs}")
    print(f"  Total images: {total_images}")
    print(f"  Complete: {total_complete} ({overall_rate:.1f}%)")
    print(f"  Needs processing: {total_needs}")
    
    # Completion status
    if overall_rate == 100:
        print("\n  ✅ All images processed!")
    elif overall_rate > 75:
        print(f"\n  🟡 Mostly complete ({total_needs} remaining)")
    elif overall_rate > 25:
        print(f"\n  🟠 Partial progress ({total_needs} remaining)")
    else:
        print(f"\n  ⚪ Just getting started ({total_needs} remaining)")


def main():
    parser = argparse.ArgumentParser(
        description="Check batch ingestion status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="Directories to check"
    )
    
    parser.add_argument(
        "--subdirs",
        action="store_true",
        help="Check all subdirectories of given directory"
    )
    
    parser.add_argument(
        "--show-incomplete",
        action="store_true",
        help="Show lists of incomplete files"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Export detailed report to JSON file"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only overall summary"
    )
    
    args = parser.parse_args()
    
    # Determine directories to check
    directories: List[Path] = []
    
    if len(args.directories) == 1 and args.subdirs:
        root = ing.expand_directory(args.directories[0])
        if not root.exists() or not root.is_dir():
            print(f"❌ Error: Not a directory: {root}")
            return 1
        directories = sorted([p for p in root.iterdir() if p.is_dir()])
        print(f"Found {len(directories)} subdirectories in {root}")
    else:
        directories = [ing.expand_directory(d) for d in args.directories]
    
    if not directories:
        print("❌ Error: No directories to check")
        return 1
    
    # Analyze all directories
    analyses = []
    for directory in directories:
        analysis = analyze_directory(directory)
        analyses.append(analysis)
    
    # Print results
    if not args.summary_only:
        for analysis in analyses:
            print_directory_status(analysis, args.show_incomplete)
    
    # Print summary
    if len(analyses) > 1 or args.summary_only:
        print_summary(analyses)
    
    # Export report
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "directories_checked": len(analyses),
            "total_images": sum(a.get("total_images", 0) for a in analyses),
            "total_complete": sum(a.get("complete", 0) for a in analyses),
            "directories": analyses
        }
        
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n📄 Report written to: {report_path}")
    
    # Exit code based on completion
    overall_complete = sum(a.get("complete", 0) for a in analyses)
    overall_total = sum(a.get("total_images", 0) for a in analyses)
    
    if overall_total > 0 and overall_complete < overall_total:
        return 1  # Incomplete
    return 0  # Complete or no images


if __name__ == "__main__":
    import sys
    sys.exit(main())
