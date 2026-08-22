import numpy as np
import pytest

from biasblaster.quantinuum_nexus import NexusExecutionConfig, counts_to_pm1_expectation


def test_counts_to_pm1_expectation_accepts_tuple_and_string_keys():
    assert np.isclose(counts_to_pm1_expectation({(0,): 75, (1,): 25}), 0.5)
    assert np.isclose(counts_to_pm1_expectation({"0": 10, "1": 30}), -0.5)


def test_nexus_execution_config_validates_resource_controls():
    config = NexusExecutionConfig(system_name="Helios-1E-lite", timeout=120.0)
    assert config.timeout == 120.0
    with pytest.raises(ValueError):
        NexusExecutionConfig(timeout=0.0)
    with pytest.raises(ValueError):
        NexusExecutionConfig(max_cost=-1.0)
