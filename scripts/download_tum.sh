#!/bin/bash
set -e
DATA_DIR="${1:-data/tum}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading TUM RGB-D sequences..."

if [ ! -d "rgbd_dataset_freiburg1_desk" ]; then
    echo "Downloading fr1/desk..."
    wget -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
    tar xzf rgbd_dataset_freiburg1_desk.tgz && rm rgbd_dataset_freiburg1_desk.tgz
fi

if [ ! -d "rgbd_dataset_freiburg2_xyz" ]; then
    echo "Downloading fr2/xyz..."
    wget -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
    tar xzf rgbd_dataset_freiburg2_xyz.tgz && rm rgbd_dataset_freiburg2_xyz.tgz
fi

if [ ! -d "rgbd_dataset_freiburg3_long_office_household" ]; then
    echo "Downloading fr3/office..."
    wget -q --show-progress https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
    tar xzf rgbd_dataset_freiburg3_long_office_household.tgz && rm rgbd_dataset_freiburg3_long_office_household.tgz
fi

echo "Done. Contents:"
ls -la "$DATA_DIR"
