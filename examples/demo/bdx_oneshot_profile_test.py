#!/usr/bin/env python3
"""Host-only guards for the fine and training-coarse granular profiles."""

from __future__ import annotations

import hashlib
import json
import os
import unittest

import bdx_oneshot_author as author


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINE_NKS = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")
FINE_NKA = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nka")
COARSE_NKS = os.path.join(
    REPO, "examples", "scenes", "bdx_oneshot_training_coarse.nks")
COARSE_NKA = os.path.join(
    REPO, "examples", "scenes", "bdx_oneshot_training_coarse.nka")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BdxOneShotGranularProfiles(unittest.TestCase):
    def test_fine_scene_bytes_and_profile_are_frozen(self):
        self.assertEqual(
            _sha256(FINE_NKS),
            "a3bf4ee2740d7c3ffd6d63438e4a915a8cf2abdb58948cd6c84e39dd55e03d63",
        )
        self.assertEqual(
            _sha256(FINE_NKA),
            "5d7a271add6b798be3df83f0b67682ba738f58524cb129ba2812315b10906812",
        )
        profile = author.granular_profile("fine")
        with open(FINE_NKS, encoding="utf-8") as stream:
            doc = json.load(stream)
        granular = next(m for m in doc["media"] if m["kind"] == "granular")
        self.assertAlmostEqual(granular["fluid_box"]["spacing"], profile.gravel_spacing)
        self.assertAlmostEqual(granular["mpm_fills"][0]["box"]["spacing"],
                               profile.debris_spacing)
        self.assertEqual(granular["mpm"]["substeps"], profile.substeps)
        self.assertAlmostEqual(granular["mpm"]["dx"], profile.dx)
        self.assertAlmostEqual(granular["mpm"]["loft_headroom"],
                               profile.loft_headroom)

    def test_training_coarse_profile_reduces_granular_resolution(self):
        fine = author.granular_profile("fine")
        coarse = author.granular_profile("training-coarse")
        self.assertEqual(coarse.gravel_spacing, 0.026)
        self.assertEqual(coarse.debris_spacing, 0.026)
        self.assertEqual(coarse.substeps, 7)
        self.assertEqual(coarse.dx, 0.052)
        self.assertEqual(coarse.loft_headroom, 0.15)
        self.assertGreater(coarse.gravel_spacing, fine.gravel_spacing)
        self.assertGreater(coarse.debris_spacing, fine.debris_spacing)
        self.assertLess(coarse.substeps, fine.substeps)
        self.assertLess(coarse.loft_headroom, fine.loft_headroom)

    def test_training_coarse_scene_selects_the_coarse_profile(self):
        with open(COARSE_NKS, encoding="utf-8") as stream:
            doc = json.load(stream)
        granular = next(m for m in doc["media"] if m["kind"] == "granular")
        profile = author.granular_profile("training-coarse")
        self.assertAlmostEqual(granular["fluid_box"]["spacing"],
                               profile.gravel_spacing)
        self.assertAlmostEqual(granular["mpm_fills"][0]["box"]["spacing"],
                               profile.debris_spacing)
        self.assertEqual(granular["mpm"]["substeps"], profile.substeps)
        self.assertAlmostEqual(granular["mpm"]["dx"], profile.dx)
        self.assertAlmostEqual(granular["mpm"]["loft_headroom"],
                               profile.loft_headroom)
        self.assertEqual(_sha256(COARSE_NKA), _sha256(FINE_NKA))

    def test_performance_probe_reports_scene_geometry_and_sample_statistics(self):
        import bdx_oneshot_perf as perf

        fine = perf.scene_geometry(FINE_NKS)
        coarse = perf.scene_geometry(COARSE_NKS)
        self.assertEqual(fine["zone_b_particles_per_env"], 9600)
        self.assertEqual(fine["zone_c_particles_per_env"], 1860)
        self.assertEqual(fine["grid_dims"], [89, 35, 53])
        self.assertEqual(coarse["zone_b_particles_per_env"], 1200)
        self.assertEqual(coarse["zone_c_particles_per_env"], 546)
        self.assertEqual(coarse["grid_dims"], [49, 22, 15])
        stats = perf.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["mean_ms"], 2.5)
        self.assertEqual(stats["median_ms"], 2.5)
        self.assertAlmostEqual(stats["p95_ms"], 3.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
