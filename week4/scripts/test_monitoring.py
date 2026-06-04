import pytest
import pandas as pd
import numpy as np
from metric_template import MetricComputer

BASELINE_PATH = '../data/demand_enriched_baseline.parquet'
NEW_DATA_PATH = '../data/demand_enriched_week4.parquet'
FEB_START = pd.Timestamp('2026-02-02')
FEB_END = pd.Timestamp('2026-02-28')

@pytest.fixture
def baseline():
    return pd.read_parquet(BASELINE_PATH)

@pytest.fixture
def new_data():
    df = pd.read_parquet(NEW_DATA_PATH)
    return df[(df['time_bucket'] >= FEB_START) & (df['time_bucket'] <= FEB_END)]

@pytest.fixture
def computer(baseline):
    return MetricComputer(baseline)


class TestNullRates:
    def test_baseline_has_no_nulls(self, baseline, computer):
        result = computer.metric_3_null_rates(baseline)
        for col, rate in result.items():
            assert rate == 0.0, f'{col} has nulls in baseline'

    def test_new_data_has_no_nulls(self, new_data, computer):
        result = computer.metric_3_null_rates(new_data)
        for col, rate in result.items():
            assert rate < 0.01, f'{col} null rate too high: {rate}'


class TestKSTest:
    def test_baseline_vs_itself_no_drift(self, baseline, computer):
        result = computer.metric_4_ks_test(baseline)
        assert not result['drift_detected'], 'Baseline vs itself should not show drift'

    def test_new_data_drift_detected(self, new_data, computer):
        result = computer.metric_4_ks_test(new_data)
        assert result['drift_detected'], 'New data should show drift vs baseline'
        assert result['p_value'] < 0.05


class TestPSI:
    def test_baseline_psi_near_zero(self, baseline, computer):
        result = computer.metric_5_psi(baseline)
        assert result < 0.05, f'PSI should be near 0 for baseline vs itself: {result}'

    def test_new_data_psi(self, new_data, computer):
        result = computer.metric_5_psi(new_data)
        assert isinstance(result, float)


class TestDuplicateRate:
    def test_baseline_no_duplicates(self, baseline, computer):
        result = computer.metric_8_duplicate_rate(baseline)
        assert result['count'] == 0
        assert not result['alert']

    def test_new_data_no_duplicates(self, new_data, computer):
        result = computer.metric_8_duplicate_rate(new_data)
        assert result['count'] == 0


class TestAccuracyByZone:
    def test_returns_dict(self, new_data, computer):
        result = computer.metric_2_accuracy_by_zone(new_data)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_accuracy_values_in_range(self, new_data, computer):
        result = computer.metric_2_accuracy_by_zone(new_data)
        for zone, acc in result.items():
            assert -1.0 <= acc <= 1.0, f'Zone {zone} accuracy out of range: {acc}'


class TestComputeAll:
    def test_compute_all_returns_all_keys(self, new_data, computer):
        result = computer.compute_all_metrics(new_data)
        expected_keys = ['accuracy', 'accuracy_by_zone', 'null_rates',
                        'ks_test', 'psi', 'prediction_distribution',
                        'data_freshness', 'duplicate_rate']
        for key in expected_keys:
            assert key in result, f'Missing key: {key}'
