import pandas as pd
import numpy as np
from typing import Dict, List
import sys

CUTOFF = pd.Timestamp('2026-01-16')
BASELINE_HOLIDAY_RATE = 0.039
HOLIDAY_RATE_THRESHOLD = 0.08
MAX_TRIP_COUNT = 500
DUPLICATE_THRESHOLD = 0

class DataQualityValidator:
    def __init__(self, baseline_df: pd.DataFrame = None):
        self.baseline = baseline_df
        self.issues = []

    def validate(self, df: pd.DataFrame) -> Dict:
        self.issues = []
        self.check_negative_trip_counts(df)
        self.check_extreme_outliers(df)
        self.check_duplicates(df)
        self.check_holiday_rate(df)
        return {
            'is_valid': len(self.issues) == 0,
            'num_issues': len(self.issues),
            'issues': self.issues,
        }

    def check_negative_trip_counts(self, df: pd.DataFrame):
        negative = (df['trip_count'] < 0).sum()
        if negative > 0:
            self._add_issue(
                issue_type='negative_trip_count',
                severity='critical',
                description=f'{negative} rows have negative trip_count values. Trip counts cannot be negative.',
                count=int(negative)
            )

    def check_extreme_outliers(self, df: pd.DataFrame):
        outliers = (df['trip_count'] > MAX_TRIP_COUNT).sum()
        if outliers > 0:
            max_val = df['trip_count'].max()
            self._add_issue(
                issue_type='extreme_outliers',
                severity='high',
                description=f'{outliers} rows have trip_count > {MAX_TRIP_COUNT} (max seen: {max_val}). Baseline max was 310.',
                count=int(outliers)
            )

    def check_duplicates(self, df: pd.DataFrame):
        dupes = df.duplicated().sum()
        if dupes > DUPLICATE_THRESHOLD:
            self._add_issue(
                issue_type='duplicate_rows',
                severity='high',
                description=f'{dupes} duplicate rows detected. Duplicates inflate demand averages and skew model training.',
                count=int(dupes)
            )

    def check_holiday_rate(self, df: pd.DataFrame):
        rate = df['is_holiday'].mean()
        if rate > HOLIDAY_RATE_THRESHOLD:
            self._add_issue(
                issue_type='holiday_rate_inflation',
                severity='medium',
                description=f'Holiday rate is {rate:.1%}, expected ~{BASELINE_HOLIDAY_RATE:.1%}. Inflated holiday flags skew demand patterns.',
                count=int(df['is_holiday'].sum())
            )

    def _add_issue(self, issue_type: str, severity: str, description: str, count: int = None, **details):
        issue = {
            'type': issue_type,
            'severity': severity,
            'description': description,
            'count': count,
            **details
        }
        self.issues.append(issue)


def main():
    print('Running data quality validation...')
    df = pd.read_parquet('data/demand_enriched_corrupted.parquet')
    baseline = df[df['time_bucket'] < CUTOFF]
    corrupted = df[df['time_bucket'] >= CUTOFF]

    print(f'Baseline rows: {len(baseline)}')
    print(f'Corrupted rows: {len(corrupted)}')

    validator = DataQualityValidator(baseline_df=baseline)
    result = validator.validate(corrupted)

    if result['is_valid']:
        print('Validation passed - no issues found.')
        sys.exit(0)
    else:
        print(f'Validation FAILED - {result["num_issues"]} issue(s) found:')
        for issue in result['issues']:
            print(f'  [{issue["severity"].upper()}] {issue["type"]}: {issue["description"]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
