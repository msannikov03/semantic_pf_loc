#!/bin/bash
set -e
DATA_DIR="${1:-data/replica}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading Replica dataset (NICE-SLAM format)..."

if [ ! -d "room0" ]; then
    echo "Downloading from NICE-SLAM hosting..."
    wget -q --show-progress https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
    echo "Extracting..."
    unzip -q Replica.zip
    if [ -d "Replica" ]; then
        mv Replica/* . && rmdir Replica
    fi
    rm -f Replica.zip
fi

echo "Done. Contents:"
ls -la "$DATA_DIR"
for scene in room0 room1 office0; do
    if [ -d "$scene" ]; then
        echo "  $scene: $(ls $scene/results/ 2>/dev/null | wc -l) frames"
    fi
done
