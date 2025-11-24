#!/usr/bin/env python3
"""
Fix YOLO image paths to point to actual image locations.
"""

from pathlib import Path

DATA_ROOT = Path("/home/mohanganesh/retail-shelf-detection/data/SKU110K")
YOLO_DIR = DATA_ROOT / "yolo_format"
IMG_DIR = DATA_ROOT / "raw" / "SKU110K_fixed" / "images"

for split in ['train', 'val', 'test', 'unlabeled']:
    txt_file = YOLO_DIR / f"{split}.txt"
    if not txt_file.exists():
        continue
    
    print(f"Fixing {split}.txt...")
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines =[]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Extract just the filename
        filename = Path(line).name        
        # Create correct path
        correct_path = IMG_DIR / filename
        
        fixed_lines.append(str(correct_path) + '\n')
    
    # Write fixed paths
    with open(txt_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"  ✓ Fixed {len(fixed_lines)} paths")

print("\n✅ All YOLO paths fixed!")
