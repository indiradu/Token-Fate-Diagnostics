import unittest

from analyze_phaselock_online_screen import summarize


class OnlineAnalysisTests(unittest.TestCase):
    def records(self):
        rows = []
        for method, elapsed, correct in (
            ("dense", 10., [1, 0]), ("surelock", 5., [1, 0]),
            ("tracelock", 6., [1, 1]), ("h2", 4., [1, 1]),
        ):
            for repeat in (-1, 0, 1, 2):
                rows.append(dict(method=method, dataset="gsm8k", batch_id=0,
                                 repeat=repeat, online_latency_s=elapsed, nfe=100,
                                 example_ids=["a", "b"], correct=correct,
                                 deterministic_vs_first=True))
        return rows

    def test_repeats_do_not_inflate_accuracy_sample_size(self):
        result = summarize(self.records(), draws=20)["methods"]["h2"]
        self.assertEqual(result["examples"], 2)
        self.assertEqual(result["sum_median_batch_seconds"], 4.)
        self.assertEqual(result["comparisons"]["surelock"]["improvements"], 1)
        self.assertAlmostEqual(result["comparisons"]["surelock"]["runtime_reduction"], .2)
        self.assertEqual(result["comparisons"]["tracelock"]["accuracy_difference_pp"], 0.)

    def test_unpaired_timing_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.records()[:-1], draws=20)

    def test_nondeterminism_is_reported(self):
        rows = self.records()
        rows[-1]["deterministic_vs_first"] = False
        result = summarize(rows, draws=20)
        self.assertFalse(result["methods"]["h2"]["deterministic"])
        self.assertIsNone(result["selection"])


if __name__ == "__main__":
    unittest.main()
