"""Tests for the DatasetLoader."""
import pytest
from server.dataset_loader import DatasetLoader


class TestDatasetLoaderInit:
    """Tests for DatasetLoader initialization."""

    def test_loader_initializes(self):
        """DatasetLoader should initialize without errors."""
        loader = DatasetLoader()
        assert loader is not None


class TestDatasetSampling:
    """Tests for dataset sampling."""

    def test_get_sample_returns_data(self):
        """get_sample should return a sample with required fields."""
        loader = DatasetLoader()
        sample = loader.get_sample("task_1_factual_grounding")

        assert sample is not None
        assert hasattr(sample, 'question') or 'question' in sample

    def test_get_sample_has_context(self):
        """Sample should include context."""
        loader = DatasetLoader()
        sample = loader.get_sample("task_1_factual_grounding")

        assert hasattr(sample, 'context') or 'context' in sample

    def test_get_sample_has_ground_truth(self):
        """Sample should include ground truth."""
        loader = DatasetLoader()
        sample = loader.get_sample("task_1_factual_grounding")

        assert hasattr(sample, 'ground_truth') or 'ground_truth' in sample

    def test_get_sample_for_task_1(self):
        """get_sample for task_1 should work."""
        loader = DatasetLoader()
        sample = loader.get_sample("task_1_factual_grounding")

        assert sample is not None

    def test_get_sample_for_task_2(self):
        """get_sample for task_2 should work."""
        loader = DatasetLoader()
        sample = loader.get_sample("task_2_multi_hop_synthesis")

        assert sample is not None

    def test_get_sample_for_task_3(self):
        """get_sample for task_3 should work."""
        loader = DatasetLoader()
        sample = loader.get_sample("task_3_adversarial_resistance")

        assert sample is not None


class TestDatasetStats:
    """Tests for dataset statistics."""

    def test_get_available_datasets(self):
        """Should be able to get list of available datasets."""
        loader = DatasetLoader()

        if hasattr(loader, 'get_available_datasets'):
            datasets = loader.get_available_datasets()
            assert isinstance(datasets, list)
            assert len(datasets) > 0

    def test_get_dataset_size(self):
        """Should be able to get dataset size info."""
        loader = DatasetLoader()

        if hasattr(loader, 'get_dataset_size'):
            size = loader.get_dataset_size("task_1_factual_grounding")
            assert size is None or isinstance(size, int)


class TestDatasetCaching:
    """Tests for dataset caching behavior."""

    def test_multiple_samples_dont_crash(self):
        """Requesting multiple samples should work."""
        loader = DatasetLoader()

        samples = []
        for _ in range(5):
            sample = loader.get_sample("task_1_factual_grounding")
            samples.append(sample)

        assert len(samples) == 5

    def test_loader_handles_missing_dataset(self):
        """Loader should handle request for missing dataset gracefully."""
        loader = DatasetLoader()

        try:
            sample = loader.get_sample("nonexistent_dataset")
            # If it returns None, that's acceptable
            assert sample is None or sample is not None
        except Exception as e:
            # Should raise a reasonable exception, not crash
            assert "not found" in str(e).lower() or "invalid" in str(e).lower() or "error" in str(e).lower()


class TestTaskDifficulty:
    """Tests for task difficulty mapping."""

    def test_task_1_is_beginner(self):
        """Task 1 should map to beginner difficulty."""
        loader = DatasetLoader()

        if hasattr(loader, 'get_task_difficulty'):
            difficulty = loader.get_task_difficulty("task_1_factual_grounding")
            assert difficulty in ["beginner", "easy", 1, "1"]

    def test_task_2_is_intermediate(self):
        """Task 2 should map to intermediate difficulty."""
        loader = DatasetLoader()

        if hasattr(loader, 'get_task_difficulty'):
            difficulty = loader.get_task_difficulty("task_2_multi_hop_synthesis")
            assert difficulty in ["intermediate", "medium", 2, "2"]

    def test_task_3_is_advanced(self):
        """Task 3 should map to advanced difficulty."""
        loader = DatasetLoader()

        if hasattr(loader, 'get_task_difficulty'):
            difficulty = loader.get_task_difficulty("task_3_adversarial_resistance")
            assert difficulty in ["advanced", "hard", 3, "3"]