import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

class MetricComputer:
    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df

    def metric_1_accuracy(self, new_df, predictions=None, actuals=None):
        if predictions is not None and actuals is not None:
            return float(np.mean(np.round(predictions) == actuals))
        baseline_mean = self.baseline_df['trip_count'].mean()
        new_mean = new_df['trip_count'].mean()
        shift_pct = abs(new_mean - baseline_mean) / baseline_mean
        return float(1.0 - shift_pct)

    def metric_2_accuracy_by_zone(self, new_df, predictions=None, actuals=None):
        results = {}
        baseline_zone_means = self.baseline_df.groupby('PULocationID')['trip_count'].mean()
        new_zone_means = new_df.groupby('PULocationID')['trip_count'].mean()
        for zone in baseline_zone_means.index:
            if zone in new_zone_means.index:
                b = baseline_zone_means[zone]
                n = new_zone_means[zone]
                shift = abs(n - b) / (b + 1e-6)
                results[int(zone)] = round(float(1.0 - shift), 4)
        return results

    def metric_3_null_rates(self, new_df):
        critical = ['trip_count', 'PULocationID', 'lag_15min', 'lag_1h', 'lag_1day']
        return {col: float(new_df[col].isna().mean()) for col in critical if col in new_df.columns}

    def metric_4_ks_test(self, new_df):
        stat, p_val = ks_2samp(self.baseline_df['trip_count'], new_df['trip_count'])
        return {
            'statistic': round(float(stat), 4),
            'p_value': round(float(p_val), 6),
            'drift_detected': bool(p_val < 0.05)
        }

    def metric_5_psi(self, new_df, bins=10):
        baseline_hist, edges = np.histogram(self.baseline_df['trip_count'], bins=bins)
        new_hist, _ = np.histogram(new_df['trip_count'], bins=edges)
        b_pct = baseline_hist / len(self.baseline_df) + 1e-6
        n_pct = new_hist / len(new_df) + 1e-6
        psi = float(np.sum((n_pct - b_pct) * np.log(n_pct / b_pct)))
        return round(psi, 4)

    def metric_6_prediction_distribution(self, predictions=None):
        if predictions is None:
            predictions = self.baseline_df['trip_count'].values
        return {
            'mean': round(float(np.mean(predictions)), 4),
            'std': round(float(np.std(predictions)), 4),
            'collapsed': bool(np.std(predictions) < 1.0)
        }

    def metric_7_data_freshness(self, new_df):
        latest = pd.to_datetime(new_df['time_bucket']).max()
        now = pd.Timestamp.now()
        age_minutes = (now - latest).total_seconds() / 60
        return {
            'latest_record': str(latest),
            'age_minutes': round(age_minutes, 1),
            'age_hours': round(age_minutes / 60, 2),
            'stale': bool(age_minutes > 120)
        }

    def metric_8_duplicate_rate(self, new_df):
        dupes = new_df.duplicated().sum()
        rate = dupes / len(new_df)
        return {
            'count': int(dupes),
            'rate': round(float(rate), 4),
            'alert': bool(rate > 0.005)
        }

    def compute_all_metrics(self, new_df, predictions=None, actuals=None):
        return {
            'accuracy': self.metric_1_accuracy(new_df, predictions, actuals),
            'accuracy_by_zone': self.metric_2_accuracy_by_zone(new_df, predictions, actuals),
            'null_rates': self.metric_3_null_rates(new_df),
            'ks_test': self.metric_4_ks_test(new_df),
            'psi': self.metric_5_psi(new_df),
            'prediction_distribution': self.metric_6_prediction_distribution(predictions),
            'data_freshness': self.metric_7_data_freshness(new_df),
            'duplicate_rate': self.metric_8_duplicate_rate(new_df),
        }
