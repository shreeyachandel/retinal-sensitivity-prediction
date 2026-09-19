from __future__ import annotations

import unittest

import pandas as pd

from retinal_sensitivity.data import DemoConfig, generate_synthetic_cohort
from retinal_sensitivity.evaluation import metric_summary
from retinal_sensitivity.modeling import grouped_predictions


class PipelineTests(unittest.TestCase):
    def test_synthetic_data_is_deterministic(self) -> None:
        config = DemoConfig(participants=10, random_state=7)
        first = generate_synthetic_cohort(config)
        second = generate_synthetic_cohort(config)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["participant_id"].str.startswith("SYN-").all())

    def test_each_participant_is_held_out_in_one_fold(self) -> None:
        data = generate_synthetic_cohort(DemoConfig(participants=10, random_state=3))
        predictions = grouped_predictions(data, n_splits=5)

        fold_counts = predictions.groupby(["model", "participant_id"])["fold"].nunique()
        self.assertTrue((fold_counts == 1).all())
        self.assertTrue(predictions["predicted_db"].notna().all())

    def test_metrics_cover_each_model(self) -> None:
        data = generate_synthetic_cohort(DemoConfig(participants=10, random_state=11))
        predictions = grouped_predictions(data, n_splits=5)
        metrics = metric_summary(predictions)

        self.assertEqual(len(metrics), predictions["model"].nunique())
        self.assertTrue((metrics["mae_db"] >= 0).all())
        self.assertTrue((metrics["rmse_db"] >= metrics["mae_db"]).all())


if __name__ == "__main__":
    unittest.main()
