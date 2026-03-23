#!/bin/bash
cd /home/anywherevla/semantic_pf_loc
source .env

echo "=== Training room0 with depth supervision ==="
python3 scripts/train_gs.py configs/train_gs/replica_room0.yaml --output_dir checkpoints_depth --depth_weight 0.5 --max_steps 30000 2>&1 | tail -5
echo ""
echo "=== Training room1 with depth supervision ==="
python3 scripts/train_gs.py configs/train_gs/replica_room1.yaml --output_dir checkpoints_depth --depth_weight 0.5 --max_steps 30000 2>&1 | tail -5
echo ""
echo "=== All done ==="
