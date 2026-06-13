#!/bin/bash
# G2 regression set (run under the GPU lock by the caller):
#  - nuka_batched_unified_world_test (21)  -- compiles against the modified hpp/cpp
#  - nuka_batched_h1_union_world_test (13) -- ditto
#  - A2 python binding smoke (GraspWorld unbroken)
set -u
cd /root/Nuka-Physics
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64
L=/root/Nuka-Physics/.g2_logs

./build-cuda128/tests/nuka_batched_unified_world_test > $L/regr_batched_unified.log 2>&1
echo "EXIT_BATCHED_UNIFIED=$?" >> $L/regr_batched_unified.log

./build-cuda128/tests/nuka_batched_h1_union_world_test > $L/regr_union13.log 2>&1
echo "EXIT_UNION13=$?" >> $L/regr_union13.log

source /root/miniconda3/etc/profile.d/conda.sh
conda activate nuka-v03
python -u python/tests/test_batched_grasp_binding.py > $L/regr_a2_binding.log 2>&1
echo "EXIT_A2_BINDING=$?" >> $L/regr_a2_binding.log

echo REGRESSIONS_DONE
