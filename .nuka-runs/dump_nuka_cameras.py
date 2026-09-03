"""Dump Nuka's actual camera matrices during a rollout step."""

import sys

sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/python")

import numpy as np
import torch

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = (
    "/mnt/c/Softwares/code/Nuka-Physics/"
    ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
)


def mat_to_quat(mat):
    """Return (w, x, y, z) from a 3x3 rotation matrix."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / s,
                (mat[2, 1] - mat[1, 2]) * s,
                (mat[0, 2] - mat[2, 0]) * s,
                (mat[1, 0] - mat[0, 1]) * s,
            ]
        )
    idx = int(np.argmax([mat[0, 0], mat[1, 1], mat[2, 2]]))
    if idx == 0:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        return np.array(
            [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ]
        )
    if idx == 1:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        return np.array(
            [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ]
        )
    s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
    return np.array(
        [
            (mat[1, 0] - mat[0, 1]) / s,
            (mat[0, 2] + mat[2, 0]) / s,
            (mat[1, 2] + mat[2, 1]) / s,
            0.25 * s,
        ]
    )


with nuka.Device.create(0) as device:
    controller = LiberoBlackBowlController(
        SCENE, device, control_backend="osc", render_quality="preview"
    )
    action = torch.tensor([0, 0, 0, 0, 0, 0, -1.0], device=controller.q.device)
    for _ in range(100):
        controller.step(action)

    controller.attach_policy_cameras()
    controller.world.step()

    # Read link7 world pose.
    link_pose = torch.from_dlpack(
        controller.world.buffer_view(nuka.ARTICULATION_LINK_POSE)
    ).view(-1, 7)[7]
    link_pos = link_pose[:3].cpu().numpy()
    link_quat = link_pose[3:].cpu().numpy()

    # Convert link quat to rotation matrix.
    w, x, y, z = link_quat
    link_mat = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    # Get camera world matrices from Nuka.
    sensor_count = controller.world.sensor_count
    print(f"sensor_count = {sensor_count}")

    for sid in range(sensor_count):
        info = controller.world.sensor_info(sid)
        print(f"\n=== sensor {sid} ===")
        print(f"mount={info.mount} index={info.index} fov={info.fov_degrees}")

        cam_pose = (
            torch.from_dlpack(controller.world.buffer_view(nuka.SENSOR_WORLD_POSE))
            .view(-1, 7)[sid]
            .cpu()
            .numpy()
        )
        cam_pos_world = cam_pose[:3]
        cam_quat_world = cam_pose[3:]

        # Convert cam quat to rotation matrix.
        w, x, y, z = cam_quat_world
        cam_mat_world = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

        print("nuka world pos :", np.round(cam_pos_world, 6).tolist())
        print("nuka world quat:", np.round(cam_quat_world, 6).tolist())
        print("nuka forward (-Z):", np.round(-cam_mat_world[:, 2], 6).tolist())
        print("nuka up      (+Y):", np.round(cam_mat_world[:, 1], 6).tolist())

        # If this is the wrist cam (mount=LINK index=7), compute local pose.
        if info.mount == nuka.SensorMount.LINK.value and info.index == 7:
            print("\n=== eye_in_hand local to link7 ===")
            print("link7 world pos :", np.round(link_pos, 6).tolist())
            print("link7 world quat:", np.round(link_quat, 6).tolist())
            local_pos = link_mat.T @ (cam_pos_world - link_pos)
            local_mat = link_mat.T @ cam_mat_world
            local_quat = mat_to_quat(local_mat)
            print("computed local pos :", np.round(local_pos, 8).tolist())
            print("computed local quat:", np.round(local_quat, 8).tolist())

    controller.close()
