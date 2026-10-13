#!/usr/bin/env python3

import unittest

import torch

from run_dual_trajectory_probe import (
    aggregate_maps,
    attention_mask,
    gaussian_blur,
    srm_residuals,
    ssim,
)


class DualTrajectoryHelpersTest(unittest.TestCase):
    def test_identical_srm_residuals_have_unit_ssim(self):
        image = torch.rand(1, 3, 32, 32)
        residual = srm_residuals(image)
        self.assertAlmostEqual(ssim(residual, residual), 1.0, places=5)

    def test_gaussian_blur_preserves_shape_and_reduces_impulse(self):
        image = torch.zeros(1, 3, 17, 17)
        image[:, :, 8, 8] = 1
        blurred = gaussian_blur(image, 1.0)
        self.assertEqual(tuple(blurred.shape), tuple(image.shape))
        self.assertLess(float(blurred.max()), 1.0)
        self.assertGreater(float(blurred.sum()), 2.9)

    def test_attention_threshold_selects_above_average_key(self):
        attention = torch.full((2, 4, 4), 0.25)
        attention[:, :, 0] = 0.55
        attention[:, :, 1:] = 0.15
        mask = attention_mask(attention, 1, 2, (4, 4), 1.3)
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        self.assertGreater(float(mask.sum()), 0)
        self.assertLess(float(mask.sum()), 16)

    def test_aggregation_starts_at_minimum_ssim_timestep(self):
        maps = [torch.full((2, 2), value) for value in (4.0, 3.0, 2.0, 1.0)]
        score, timestep, minimum = aggregate_maps(
            [3, 2, 1, 0], maps, [0.8, 0.2, 0.4, 0.7]
        )
        self.assertEqual(timestep, 2)
        self.assertEqual(minimum, 0.2)
        self.assertTrue(torch.allclose(score, torch.full((2, 2), 2.0)))


if __name__ == "__main__":
    unittest.main()
