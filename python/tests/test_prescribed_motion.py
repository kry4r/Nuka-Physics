"""Analytic twist and geometric error contracts independent of solver iteration counts."""

import numpy as np

from nuka.geometry import cone_mesh
from nuka.motion import JointMotion


def test_joint_twists_include_child_frame_and_com_offsets():
    root = [0.1, -0.2, 0.3, np.cos(.3), 0, np.sin(.3), 0]
    joints = [
        dict(name="rotation", parent=-1, child=0, type="revolute", parent_frame=root,
             child_frame=[.1, -.2, .05, 1, 0, 0, 0], axis=[1, 2, 3]),
        dict(name="translation", parent=0, child=1, type="prismatic",
             parent_frame=[.2, .3, -.1, np.cos(.2), np.sin(.2), 0, 0],
             child_frame=[-.1, 0, .07, 1, 0, 0, 0], axis=[0, 1, 0]),
    ]
    com = np.array([[.02, -.01, .04], [-.03, .04, .01]])
    motion = JointMotion(joints, [[0, 0, 0, 1, 0, 0, 0]] * 2, com)
    q = dict(rotation=np.array([.2, -.7]), translation=np.array([.4, -.2]))
    qd = dict(rotation=np.array([.3, -.2]), translation=np.array([-.2, .5]))
    pose, linear, angular = motion.evaluate(q, qd)
    delta = 1e-6
    plus = motion.evaluate({k: q[k] + delta * qd[k] for k in q}, qd)[0]
    minus = motion.evaluate({k: q[k] - delta * qd[k] for k in q}, qd)[0]

    def rotate(rotation, value):
        t = 2 * np.cross(rotation[..., 1:], value)
        return value + rotation[..., :1] * t + np.cross(rotation[..., 1:], t)

    center_plus = plus[..., :3] + rotate(plus[..., 3:], com)
    center_minus = minus[..., :3] + rotate(minus[..., 3:], com)
    np.testing.assert_allclose(linear, (center_plus - center_minus) / (2 * delta), atol=2e-9, rtol=1e-8)
    axis = np.broadcast_to([.2, -.3, .7], linear.shape)
    derivative = (rotate(plus[..., 3:], axis) - rotate(minus[..., 3:], axis)) / (2 * delta)
    np.testing.assert_allclose(derivative, np.cross(angular, rotate(pose[..., 3:], axis)), atol=2e-9, rtol=1e-8)


def test_cone_support_bound_and_closed_outward_surface():
    radius, height, tolerance = .012, .02, 1e-5
    mesh = cone_mesh(radius, height, max_error=tolerance)
    assert mesh.error_bound <= tolerance
    direction = np.random.default_rng(1).normal(size=(5000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    exact = np.maximum(radius * np.linalg.norm(direction[:, :2], axis=1) - height * direction[:, 2],
                       height * direction[:, 2])
    polygon = (direction @ mesh.vertices.T).max(axis=1)
    assert (exact - polygon).min() >= -1e-16
    assert (exact - polygon).max() <= mesh.error_bound + 1e-16
    faces = mesh.vertices[mesh.triangles]
    normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    assert np.all(np.linalg.norm(normal, axis=1) > 0)
    assert np.all(np.sum(normal * (faces.mean(axis=1) - [0, 0, -height / 2]), axis=1) > 0)
    edges = np.concatenate([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
