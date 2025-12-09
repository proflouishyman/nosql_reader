#!/usr/bin/env python3
"""
migrate_borr_data.py - Migrate BORR data files to archives directory

This script:
1. Copies JSON/TXT files from source to archives
2. Copies matching image files
3. Renames .txt files to .json for proper ingestion
4. Preserves directory structure
5. Handles filename mismatches (e.g., _jpg.txt vs .jpg)

Usage:
    python migrate_borr_data.py --dry-run    # Preview changes
    python migrate_borr_data.py              # Run actual migration
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# ============================================================================

SOURCE_PATH = "/Users/louishyman/coding/Larger Sample BORR Data"
DEST_PATH = "/Users/louishyman/coding/nosql/archives/borr_data"

# ============================================================================


class BORRMigrator:
    """Migrate BORR historical documents to archives directory."""
    
    def __init__(self, source: str, dest: str, dry_run: bool = False):
        self.source = Path(source).resolve()
        self.dest = Path(dest).resolve()
        self.dry_run = dry_run
        
        self.stats = {
            'json_files_found': 0,
            'json_files_copied': 0,
            'images_found': 0,
            'images_copied': 0,
            'mismatches': 0,
            'errors': []
        }
    
    def find_image_for_json(self, json_path: Path) -> Optional[Path]:
        """
        Find the matching image file for a JSON/TXT file.
        
        Handles these naming patterns:
        - RDApp-205406Porter001_jpg.txt → RDApp-205406Porter001.jpg
        - roll_1-2322.jpg.txt → roll_1-2322.jpg  
        - page_001.png.json → page_001.png
        """
        # Get the base filename without final extension (.txt or .json)
        filename = json_path.stem
        
        image_candidates = []
        
        # Pattern 1: Check if stem already has image extension (microfilm pattern)
        # e.g., roll_1-2322.jpg.txt → stem is "roll_1-2322.jpg"
        if re.search(r'\.(jpg|jpeg|png|tif|tiff|bmp|gif)$', filename, re.IGNORECASE):
            # Stem already contains image extension - use it directly
            image_candidates.append(json_path.parent / filename)
        
        # Pattern 2: filename_ext.txt → filename.ext (Relief Records pattern)
        # e.g., RDApp-205406Porter001_jpg.txt → RDApp-205406Porter001.jpg
        match = re.match(r'(.+)_(jpg|jpeg|png|tif|tiff)', filename, re.IGNORECASE)
        if match:
            base_name = match.group(1)
            ext = match.group(2)
            image_candidates.extend([
                json_path.parent / f"{base_name}.{ext}",
                json_path.parent / f"{base_name}.{ext.upper()}",
                json_path.parent / f"{base_name}.{ext.lower()}"
            ])
        
        # Pattern 3: Generic fallback - try common extensions
        # Strip .json/.txt from base_name if present
        base_name = filename.replace('.json', '').replace('.txt', '')
        # Only add these if we haven't already found candidates
        if not image_candidates:
            image_candidates.extend([
                json_path.parent / f"{base_name}.jpg",
                json_path.parent / f"{base_name}.jpeg",
                json_path.parent / f"{base_name}.png",
                json_path.parent / f"{base_name}.tif",
                json_path.parent / f"{base_name}.tiff",
                json_path.parent / f"{base_name}.JPG",
                json_path.parent / f"{base_name}.JPEG",
                json_path.parent / f"{base_name}.PNG",
            ])
        
        # Return first existing candidate
        for candidate in image_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        
        return None
    
    def normalize_json_content(self, json_path: Path, relative_dest_path: str) -> Dict:
        """
        Load JSON content and normalize it for the archives.
        
        - Ensures relative_path field exists and is correct
        - Handles both .json and .txt extensions
        - Extracts person metadata from folder structure
        - Updates metadata if needed
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure relative_path is set correctly
            # Should be relative to archives root, with .json extension
            dest_json_name = json_path.stem
            if not dest_json_name.endswith('.json'):
                dest_json_name = f"{dest_json_name}.json"
            
            data['relative_path'] = str(Path(relative_dest_path) / dest_json_name)
            
            # Extract person metadata from folder structure
            # Pattern: "205406-Porter" in path
            path_parts = json_path.parts
            person_folder = None
            person_id = None
            person_name = None
            
            for part in path_parts:
                # Match pattern like "205406-Porter"
                match = re.match(r'^(\d+)-(.+)$', part)
                if match:
                    person_folder = part
                    person_id = match.group(1)
                    person_name = match.group(2)
                    break
            
            # Add person metadata
            data['person_folder'] = person_folder
            data['person_id'] = person_id
            data['person_name'] = person_name
            
            # Determine collection
            collection = None
            if 'Relief Record Scans' in str(json_path):
                collection = 'Relief Record Scans'
            elif 'Microfilm Digitization' in str(json_path):
                collection = 'Microfilm Digitization'
            
            data['collection'] = collection
            
            # Build archive structure metadata
            archive_structure = {}
            if collection == 'Relief Record Scans':
                # Extract format and box from path
                for i, part in enumerate(path_parts):
                    if part in ['JPG', 'TIF', 'TIFF', 'PNG']:
                        archive_structure['format'] = part
                    if 'Box' in part:
                        archive_structure['physical_box'] = part
            elif collection == 'Microfilm Digitization':
                # Extract series info
                for part in path_parts:
                    if 'Microfilm' in part:
                        archive_structure['series'] = part
            
            archive_structure['path_components'] = list(Path(relative_dest_path).parts)
            data['archive_structure'] = archive_structure
            
            # Add migration metadata
            if 'metadata' not in data:
                data['metadata'] = {}
            
            data['metadata']['migrated_from'] = str(json_path)
            data['metadata']['migration_date'] = None  # Will be set by ingestion
            
            return data
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")
    
    def migrate_file_pair(self, json_path: Path) -> Tuple[bool, str]:
        """
        Migrate a JSON/TXT file and its matching image.
        
        Returns:
            (success, message)
        """
        # Calculate relative path from source
        try:
            rel_path = json_path.relative_to(self.source)
        except ValueError:
            return False, f"File {json_path} is not under source directory"
        
        # Determine destination paths
        dest_json = self.dest / rel_path.parent / f"{json_path.stem}.json"
        
        # Find matching image
        image_path = self.find_image_for_json(json_path)
        
        if not image_path:
            self.stats['mismatches'] += 1
            return False, f"No matching image found for {json_path.name}"
        
        dest_image = self.dest / rel_path.parent / image_path.name
        
        # Prepare normalized JSON content
        try:
            json_data = self.normalize_json_content(json_path, str(rel_path.parent))
        except ValueError as e:
            self.stats['errors'].append(str(e))
            return False, str(e)
        
        # Execute migration (or simulate if dry-run)
        if self.dry_run:
            print(f"  [DRY-RUN] Would copy:")
            print(f"    JSON: {json_path} → {dest_json}")
            print(f"    Image: {image_path} → {dest_image}")
            self.stats['json_files_copied'] += 1
            self.stats['images_copied'] += 1
            return True, "Dry-run successful"
        
        try:
            # Create destination directories
            dest_json.parent.mkdir(parents=True, exist_ok=True)
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            shutil.copy2(image_path, dest_image)
            self.stats['images_copied'] += 1
            
            # Write normalized JSON
            with open(dest_json, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            self.stats['json_files_copied'] += 1
            
            return True, f"Migrated {json_path.name} + {image_path.name}"
        
        except Exception as e:
            error_msg = f"Error migrating {json_path.name}: {e}"
            self.stats['errors'].append(error_msg)
            return False, error_msg
    
    def migrate_all(self) -> None:
        """Migrate all JSON/TXT files and their matching images."""
        
        if not self.source.exists():
            print(f"❌ Source directory does not exist: {self.source}")
            return
        
        if not self.dry_run:
            self.dest.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Scanning source: {self.source}")
        print(f"📁 Destination: {self.dest}")
        print(f"{'🔬 DRY-RUN MODE - No files will be modified' if self.dry_run else '✍️  LIVE MODE - Files will be copied'}\n")
        
        # Find all JSON/TXT files
        json_files = []
        for ext in ['*.json', '*.txt']:
            json_files.extend(self.source.rglob(ext))
        
        self.stats['json_files_found'] = len(json_files)
        
        # Find all images
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp', '*.gif']
        image_files = []
        for ext in image_exts:
            image_files.extend(self.source.rglob(ext))
        
        self.stats['images_found'] = len(image_files)
        
        print(f"📊 Found {self.stats['json_files_found']} JSON/TXT files")
        print(f"📊 Found {self.stats['images_found']} image files\n")
        
        # Process each JSON file
        for i, json_path in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Processing {json_path.name}...")
            success, message = self.migrate_file_pair(json_path)
            
            if not success:
                print(f"  ⚠️  {message}")
            else:
                print(f"  ✅ {message}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print migration statistics."""
        print("\n" + "="*60)
        print("📊 MIGRATION SUMMARY")
        print("="*60)
        print(f"JSON/TXT files found:    {self.stats['json_files_found']}")
        print(f"JSON files migrated:     {self.stats['json_files_copied']}")
        print(f"Images found:            {self.stats['images_found']}")
        print(f"Images migrated:         {self.stats['images_copied']}")
        print(f"Mismatches (no image):   {self.stats['mismatches']}")
        print(f"Errors:                  {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\n❌ ERRORS:")
            for error in self.stats['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more")
        
        print("="*60)
        
        if self.dry_run:
            print("\n💡 This was a DRY-RUN. Run without --dry-run to perform actual migration.")
            print(f"\n   python migrate_borr_data.py")
        else:
            print(f"\n✅ Migration complete! Files are in: {self.dest}")
            print("\n📝 Next steps:")
            print(f"   1. Verify files: ls -la {self.dest}")
            print("   2. Update .env if needed (ARCHIVES_HOST_PATH should point to archives parent)")
            print("   3. Restart containers: docker compose down && docker compose up -d")
            print("   4. Ingest via UI: Settings → Data Ingestion → Scan for new images")
            print("   5. Or via command: docker compose exec app python app/data_processing.py /data/archives/borr_data")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate BORR historical documents to archives directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration (edit paths at top of script):
  SOURCE_PATH: {SOURCE_PATH}
  DEST_PATH:   {DEST_PATH}

Examples:
  # Dry-run to preview changes
  python migrate_borr_data.py --dry-run

  # Actual migration
  python migrate_borr_data.py
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without copying files'
    )
    
    args = parser.parse_args()
    
    print(f"📂 Configuration:")
    print(f"   Source: {SOURCE_PATH}")
    print(f"   Destination: {DEST_PATH}")
    print()
    
    migrator = BORRMigrator(
        source=SOURCE_PATH,
        dest=DEST_PATH,
        dry_run=args.dry_run
    )
    
    migrator.migrate_all()


if __name__ == '__main__':
    main()