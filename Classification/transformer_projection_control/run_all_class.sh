#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== Running projection control experiments ==="

# echo "--- Clone detection ---"
# python transformer_projection_control/clone_vh_projection.py --model graphcodebert
# python transformer_projection_control/clone_cg_projection.py --model graphcodebert
# python transformer_projection_control/clone_vh_cg_projection.py --model graphcodebert

echo "--- Code classification ---"
python transformer_projection_control/class_vh_projection.py --model graphcodebert
python transformer_projection_control/class_cg_projection.py --model graphcodebert
python transformer_projection_control/class_vh_cg_projection.py --model graphcodebert

echo "=== All projection experiments complete ==="
