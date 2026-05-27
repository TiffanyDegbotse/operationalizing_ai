import pytest
import pandas as pd
import numpy as np
from validation.check_data_quality import DataQualityValidator

CUTOFF = pd.Timestamp('2026-01-16')

@pytest.fixture
def baseline_data():
    df = pd.read_parquet('data/demand_enriched_corrupted.parquet')
    return df[df['time_bucket'] < CUTOFF]

@pytest.fixture
def corrupted_data():
    df = pd.read_parquet('data/demand_enriched_corrupted.parquet')
    return df[df['time_bucket'] >= CUTOFF]

@pytest.fixture
def validator(baseline_data):
    return DataQualityValidator(baseline_df=baseline_data)


class TestBaselineData:
    def test_baseline_passes_validation(self, baseline_data, validator):
        result = validator.validate(baseline_data)
        issues = result['issues']
        assert result['is_valid'], f'Baseline failed: {issues}'


class TestDataQualityIssues:
    def test_detect_negative_trip_counts(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        types = [i['type'] for i in result['issues']]
        assert 'negative_trip_count' in types

    def test_detect_extreme_outliers(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        types = [i['type'] for i in result['issues']]
        assert 'extreme_outliers' in types

    def test_detect_duplicates(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        types = [i['type'] for i in result['issues']]
        assert 'duplicate_rows' in types

    def test_detect_holiday_rate_inflation(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        types = [i['type'] for i in result['issues']]
        assert 'holiday_rate_inflation' in types

    def test_corrupted_data_fails_validation(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert not result['is_valid']

    def test_corrupted_has_four_issues(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        num = result['num_issues']
        assert num == 4, f'Expected 4 issues, got {num}'


class TestGracefulDegradation:
    def test_api_does_not_crash_with_bad_data(self, corrupted_data, validator):
        try:
            result = validator.validate(corrupted_data)
            assert 'is_valid' in result
            assert 'issues' in result
        except Exception as e:
            pytest.fail(f'Validator crashed on bad data: {e}')

    def test_validator_returns_structured_result(self, corrupted_data, validator):
        result = validator.validate(corrupted_data)
        assert isinstance(result['is_valid'], bool)
        assert isinstance(result['issues'], list)
        assert isinstance(result['num_issues'], int)
        for issue in result['issues']:
            assert 'type' in issue
            assert 'severity' in issue
            assert 'description' in issue
