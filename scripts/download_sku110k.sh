#!/bin/bash
# Download SKU-110K dataset for paper reproduction
# Paper uses 10,000 images from SKU-110K dataset

set -e  # Exit on error

echo "=========================================="
echo "Downloading SKU-110K Dataset"
echo "=========================================="
echo "Size: ~11GB"
echo "Images: 11,762 total"
echo "Paper uses: 10,000 selected images"
echo "=========================================="

cd /home/mohanganesh/retail-shelf-detection/data/SKU110K/raw

# Check if already downloaded
if [ -d "SKU110K_fixed" ]; then
    echo "✓ Dataset already exists at $(pwd)/SKU110K_fixed"
    echo "Verifying contents..."
    
    if [ -d "SKU110K_fixed/images" ]; then
        num_images=$(find SKU110K_fixed/images -name "*.jpg" | wc -l)
        echo "✓ Found $num_images images"
        
        if [ $num_images -ge 11000 ]; then
            echo "✓ Dataset appears complete"
            echo ""
            echo "If you want to re-download, delete the directory first:"
            echo "  rm -rf $(pwd)/SKU110K_fixed"
            exit 0
        else
            echo "⚠ Expected 11,762 images, found $num_images"
            echo "Re-downloading..."
            rm -rf SKU110K_fixed SKU110K_fixed.tar.gz
        fi
    fi
fi

echo ""
echo "Starting download from AWS S3..."
echo "This will take 10-30 minutes depending on connection speed"
echo ""

# Download from AWS S3 (fastest and most reliable)
wget -c http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz

echo ""
echo "Download complete! Extracting..."
echo "This will take 5-10 minutes..."
echo ""

# Extract
tar -xzf SKU110K_fixed.tar.gz

# Verify extraction
if [ -d "SKU110K_fixed" ]; then
    echo ""
    echo "=========================================="
    echo "✓ EXTRACTION COMPLETE"
    echo "=========================================="
    
    # Count images
    if [ -d "SKU110K_fixed/images" ]; then
        num_images=$(find SKU110K_fixed/images -name "*.jpg" | wc -l)
        echo "✓ Images: $num_images"
    fi
    
    # Check annotations
    if [ -d "SKU110K_fixed/annotations" ]; then
        num_annotations=$(find SKU110K_fixed/annotations -name "*.csv" | wc -l)
        echo "✓ Annotation files: $num_annotations"
    fi
    
    echo ""
    echo "Dataset structure:"
    tree -L 2 SKU110K_fixed 2>/dev/null || ls -lR SKU110K_fixed | head -20
    
    echo ""
    echo "=========================================="
    echo "✓ DOWNLOAD SUCCESSFUL"
    echo "=========================================="
    echo "Location: $(pwd)/SKU110K_fixed"
    echo ""
    echo "Next steps:"
    echo "  1. python core/dataset/create_splits.py"
    echo "  2. python core/dataset/coco_converter.py"
    echo "  3. python core/dataset/yolo_converter.py"
    echo "=========================================="
    
else
    echo "✗ Extraction failed"
    exit 1
fi
