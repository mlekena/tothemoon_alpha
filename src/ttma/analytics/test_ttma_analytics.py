import pytest
import pytest_benchmark
from ttma_analytics import fat_list


def test_sum(benchmark):
    assert len(benchmark(fat_list)) != 0