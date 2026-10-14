#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_localization import average_precision, roc_auc


class LocalizationMetricsTest(unittest.TestCase):
    def test_perfect_ranking(self):
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)

    def test_inverted_ranking(self):
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.9, 0.1, 0.8, 0.2])
        self.assertAlmostEqual(roc_auc(labels, scores), 0.0)


if __name__ == "__main__":
    unittest.main()
