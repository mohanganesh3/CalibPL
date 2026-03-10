#!/bin/bash
# Download CrowdHuman dataset via Kaggle
# Uses the kagkimch/crowdhuman-coco mirror which contains the full 10.89GB dataset
# likely with COCO formatting included.

set -e

echo "=========================================="
echo "Downloading CrowdHuman Dataset via Kaggle"
echo "=========================================="
echo "Size: ~11GB"
echo "Using Kaggle API: kagkimch/crowdhuman-coco"
echo "=========================================="

DATA_DIR="/home/mohanganesh/retail-shelf-detection/data/CrowdHuman/raw"
KAGGLE_BIN="/home/mohanganesh/.local/bin/kaggle"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -f "crowdhuman-coco.zip" ]; then
    echo "✓ Archive already downloaded: crowdhuman-coco.zip"
else
    echo "Starting download from Kaggle..."
    echo "This will take 10-30 minutes depending on connection speed"
    $KAGGLE_BIN datasets download -d kagkimch/crowdhuman-coco
fi

echo ""
echo "Download complete! Extracting..."
echo "This will take a few minutes..."

# Extract quietly
unzip -q crowdhuman-coco.zip

echo "=========================================="
echo "✓ EXTRACTION COMPLETE"
echo "=========================================="
echo "Files found:"
ls -la

# Try to find annotations
echo "Searching for annotations:"
find . -name "*.json" | head -10 || echo "Not found easily"

echo "=========================================="
echo "✓ DOWNLOAD SUCCESSFUL"
echo "=========================================="
echo "Location: $(pwd)"
