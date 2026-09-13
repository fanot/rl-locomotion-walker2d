#!/usr/bin/env bash
# Reward ablation: how the control-cost weight changes the learned gait.
#
#   bash run_ablation.sh            # 1M steps, 3 configs x 2 seeds
#   STEPS=300000 SEEDS="1" bash run_ablation.sh
#
# The three configs differ only in ctrl_cost_weight, so the returns are NOT
# directly comparable across configs -- compare on x_distance / action_energy,
# which do not depend on the reward weights (see README).

set -euo pipefail

STEPS=${STEPS:-1000000}
SEEDS=${SEEDS:-"1 2"}
ENV_ID=${ENV_ID:-Walker2d-v5}

for seed in $SEEDS; do
  for cfg in "ctrl0:0.0" "ctrl_default:0.001" "ctrl_high:0.01"; do
    name="${cfg%%:*}"
    weight="${cfg##*:}"
    echo "=== ${name} (ctrl_cost_weight=${weight}) seed ${seed} ==="
    python3 ppo.py \
      --env-id "$ENV_ID" \
      --total-timesteps "$STEPS" \
      --seed "$seed" \
      --ctrl-cost-weight "$weight" \
      --run-name "${name}_s${seed}"
  done
done

python3 plot.py --metric episodic_return --out curve_return.png
python3 plot.py --metric x_distance      --out curve_distance.png
python3 plot.py --metric action_energy   --out curve_energy.png
