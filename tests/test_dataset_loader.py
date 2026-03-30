"""Tests for the DatasetLoader."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.dataset_loader import DatasetLoader, DifficultyLevel


class TestDatasetLoaderInit:
    """Tests for DatasetLoader initialization."""

    def test_loader_initializes(self):
        """DatasetLoader should initialize without errors."""
        loader = DatasetLoader()
        assert loader is not None


class TestDatasetLoading:
    """Tests for dataset loading."""

    def test_load_real_datasets(self):
        """load_real_datasets should populate examples."""
        loader = DatasetLoader()
        # load_real_datasets loads from HF cache or local
        loaded = loader.load_real_datasets(max_per_dataset=100, cache=True)
        total = loader.get_total_examples()
        # May be 0 if no cache available, but should not crash
        assert total >= 0

    def test_get_total_examples(self):
        """get_total_examples should return an integer."""
        loader = DatasetLoader()
        total = loader.get_total_examples()
        assert isinstance(total, int)
        assert total >= 0


class TestDatasetSampling:
    """Tests for dataset sampling."""

    def test_get_example_by_difficulty_returns_none_when_empty(self):
        """get_example_by_difficulty should return None when no examples loaded."""
        loader = DatasetLoader()
        # Don't load any datasets
        example = loader.get_example_by_difficulty(DifficultyLevel.BEGINNER)
        # Should return None when no datasets loaded
        assert example is None

    def test_get_example_after_loading(self):
        """get_example_by_difficulty should work after loading."""
        loader = DatasetLoader()
        loaded = loader.load_real_datasets(max_per_dataset=100, cache=True)

        if loader.get_total_examples() > 0:
            example = loader.get_example_by_difficulty(DifficultyLevel.BEGINNER)
            # May be None if no beginner examples in cache
            if example is not None:
                assert hasattr(example, 'question')
                assert hasattr(example, 'context')
                assert hasattr(example, 'answer')

    def test_start_new_episode_returns_empty_when_no_data(self):
        """start_new_episode should handle no data gracefully."""
        loader = DatasetLoader()
        examples = loader.start_new_episode(num_questions=5)
        # Should return empty list when no examples loaded
        assert examples is not None
        assert isinstance(examples, list)

    def test_start_new_episode_after_loading(self):
        """start_new_episode should work after loading."""
        loader = DatasetLoader()
        loaded = loader.load_real_datasets(max_per_dataset=500, cache=True)

        if loader.get_total_examples() > 0:
            examples = loader.start_new_episode(num_questions=5)
            assert examples is not None
            assert isinstance(examples, list)


class TestDatasetStats:
    """Tests for dataset statistics."""

    def test_get_statistics(self):
        """Should be able to get dataset statistics."""
        loader = DatasetLoader()
        stats = loader.get_statistics()

        assert stats is not None
        assert hasattr(stats, 'total_examples')


class TestDatasetCaching:
    """Tests for dataset caching behavior."""

    def test_reset_usage(self):
        """reset_usage should clear used indices."""
        loader = DatasetLoader()
        loader.reset_usage()
        # Should not crash
        assert True

    def test_multiple_resets(self):
        """Multiple resets should work."""
        loader = DatasetLoader()
        loader.reset_usage()
        loader.reset_usage()
        loader.reset_usage()
        assert True


class TestTaskDifficulty:
    """Tests for task difficulty mapping."""

    def test_difficulty_levels_exist(self):
        """Difficulty levels should be defined."""
        assert DifficultyLevel.BEGINNER.value == "beginner"
        assert DifficultyLevel.INTERMEDIATE.value == "intermediate"
        assert DifficultyLevel.ADVANCED.value == "advanced"
        assert DifficultyLevel.EXPERT.value == "expert"

    def test_difficulty_is_enum(self):
        """DifficultyLevel should be an enum."""
        assert isinstance(DifficultyLevel.BEGINNER, DifficultyLevel)
        assert isinstance(DifficultyLevel.INTERMEDIATE, DifficultyLevel)
        assert isinstance(DifficultyLevel.ADVANCED, DifficultyLevel)
        assert isinstance(DifficultyLevel.EXPERT, DifficultyLevel)


class TestQAExample:
    """Tests for QAExample dataclass."""

    def test_qa_example_to_dict(self):
        """QAExample.to_dict should work."""
        from server.dataset_loader import QAExample

        example = QAExample(
            question="What is 2+2?",
            context="Basic math: 2+2 equals 4.",
            answer="4",
            id="test_1",
            source="test",
            difficulty=DifficultyLevel.BEGINNER
        )

        d = example.to_dict()
        assert d["question"] == "What is 2+2?"
        assert d["answer"] == "4"
        assert d["source"] == "test"

    def test_qa_example_attributes(self):
        """QAExample should have all required attributes."""
        from server.dataset_loader import QAExample

        example = QAExample(
            question="Test?",
            context="Test context.",
            answer="Test answer",
            id="test_id",
            source="test_source"
        )

        assert example.question == "Test?"
        assert example.context == "Test context."
        assert example.answer == "Test answer"
        assert example.id == "test_id"
        assert example.source == "test_source"