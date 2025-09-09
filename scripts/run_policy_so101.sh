#!/bin/zsh

# Default device ports and identifiers (override via env vars if needed)
ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodem5A7A0178011}
ROBOT_ID=${ROBOT_ID:-my_awesome_follower_arm}

# Policy and dataset used to shape the policy I/O
POLICY_PATH=${POLICY_PATH:-tinjyuu/my-lerobot-policy-4}
HF_USER=${HF_USER:-tinjyuu}
DATASET_REPO_ID=${DATASET_REPO_ID:-${HF_USER}/eval_so101-7}

# Control loop
FPS=${FPS:-30}
SINGLE_TASK=${SINGLE_TASK:-"Put lego brick into the transparent box"}

# Activate the conda environment 'lerobot'
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook)"
  conda activate lerobot
else
  echo "conda not found. Please install conda and create the 'lerobot' env." >&2
  exit 1
fi

python - <<'PY'
import os
import time

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import make_robot_from_config, so101_follower
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging


def main():
    init_logging()

    robot_port = os.environ.get("ROBOT_PORT")
    robot_id = os.environ.get("ROBOT_ID")
    policy_path = os.environ.get("POLICY_PATH")
    dataset_repo_id = os.environ.get("DATASET_REPO_ID")
    fps = int(os.environ.get("FPS", 30))
    single_task = os.environ.get("SINGLE_TASK", "")

    # Build robot config and instance
    robot_cfg = so101_follower.config_so101_follower.SO101FollowerConfig(
        port=robot_port,
        id=robot_id,
    )
    robot = make_robot_from_config(robot_cfg)

    # Load pretrained policy config
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)

    # Use dataset metadata to configure the policy features
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
    policy = make_policy(policy_cfg, ds_meta=ds_meta)

    robot.connect()
    try:
        while True:
            loop_start = time.perf_counter()

            observation = robot.get_observation()
            observation_frame = build_dataset_frame(ds_meta.features, observation, prefix="observation")

            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

            robot.send_action(action)

            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
PY


