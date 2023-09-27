import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "DrawBench.py"


def test_load_dataset(dataset_path: str, expected_num_test: int = 200):
    dataset = ds.load_dataset(path=dataset_path)
    assert dataset["test"].num_rows == expected_num_test
