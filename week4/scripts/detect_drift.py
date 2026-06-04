import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import sys

BASELINE_PATH = '../data/demand_enriched_baseline.parquet'
NEW_DATA_PATH = '../data/demand_enriched_week4.parquet'
FEB_START = pd.Timestamp('2026-02-02')
FEB_END = pd.Timestamp('2026-02-28')

def psi(baseline_series, new_series, bins=10):
    baseline_hist, edges = np.histogram(baseline_series, bins=bins)
    new_hist, _ = np.histogram(new_series, bins=edges)
    b_pct = baseline_hist / len(baseline_series) + 1e-6
    n_pct = new_hist / len(new_series) + 1e-6
    return float(np.sum((n_pct - b_pct) * np.log(n_pct / b_pct)))

def detect_feature_drift(baseline_df, new_df, feature):
    ks_stat, p_val = ks_2samp(baseline_df[feature], new_df[feature])
    psi_val = psi(baseline_df[feature], new_df[feature])
    return {
        'feature': feature,
        'ks_statistic': round(float(ks_stat), 4),
        'p_value': round(float(p_val), 6),
        'psi': round(psi_val, 4),
        'drift_detected': bool(p_val < 0.05),
        'baseline_mean': round(float(baseline_df[feature].mean()), 4),
        'new_mean': round(float(new_df[feature].mean()), 4),
    }

def detect_concept_drift_by_segment(baseline_df, new_df):
    results = {}
    common_zones = set(baseline_df['PULocationID'].unique()) & set(new_df['PULocationID'].unique())
    zone_drift = []
    for zone in common_zones:
        b = baseline_df[baseline_df['PULocationID'] == zone]['trip_count']
        n = new_df[new_df['PULocationID'] == zone]['trip_count']
        if len(b) > 10 and len(n) > 10:
            ks_stat, p_val = ks_2samp(b, n)
            shift = n.mean() - b.mean()
            zone_drift.append({
                'zone': int(zone),
                'ks_statistic': round(float(ks_stat), 4),
                'p_value': round(float(p_val), 6),
                'mean_shift': round(float(shift), 2),
                'drift_detected': bool(p_val < 0.05)
            })
    zone_drift.sort(key=lambda x: x['ks_statistic'], reverse=True)
    results['top_drifted_zones'] = zone_drift[:10]
    results['zones_with_drift'] = sum(1 for z in zone_drift if z['drift_detected'])
    results['total_zones'] = len(zone_drift)
    return results

def main():
    print('=' * 70)
    print('DRIFT DETECTION REPORT')
    print('=' * 70)

    baseline = pd.read_parquet(BASELINE_PATH)
    week4 = pd.read_parquet(NEW_DATA_PATH)
    new_data = week4[(week4['time_bucket'] >= FEB_START) & (week4['time_bucket'] <= FEB_END)]

    print(f'Baseline: {len(baseline)} rows (Jan 1-15 2026)')
    print(f'New data: {len(new_data)} rows (Feb 2-28 2026)')

    drift_found = False

    print()
    print('--- Pattern 1: Trip Count Distribution Drift ---')
    result = detect_feature_drift(baseline, new_data, 'trip_count')
    print(f'  KS statistic: {result["ks_statistic"]}')
    print(f'  p-value: {result["p_value"]}')
    print(f'  Baseline mean: {result["baseline_mean"]} → New mean: {result["new_mean"]}')
    if result['drift_detected']:
        print('  STATUS: DRIFT DETECTED — trip_count distribution shifted significantly')
        drift_found = True

    print()
    print('--- Pattern 2: Day-of-Week Distribution Drift ---')
    result2 = detect_feature_drift(baseline, new_data, 'dayofweek')
    print(f'  KS statistic: {result2["ks_statistic"]}')
    print(f'  p-value: {result2["p_value"]}')
    print(f'  Baseline mean: {result2["baseline_mean"]} → New mean: {result2["new_mean"]}')
    if result2['drift_detected']:
        print('  STATUS: DRIFT DETECTED — day-of-week distribution shifted')
        drift_found = True

    print()
    print('--- Pattern 3: Lag Feature Drift ---')
    for feat in ['lag_15min', 'lag_1h', 'roll_mean_1h']:
        r = detect_feature_drift(baseline, new_data, feat)
        status = 'DRIFT' if r['drift_detected'] else 'OK'
        print(f'  {feat}: KS={r["ks_statistic"]}, p={r["p_value"]} [{status}]')
        if r['drift_detected']:
            drift_found = True
    print('  STATUS: Lag features reflect underlying trip_count drop')

    print()
    print('--- Pattern 4: Zone-Level Concept Drift ---')
    segment_result = detect_concept_drift_by_segment(baseline, new_data)
    print(f'  Zones with drift: {segment_result["zones_with_drift"]}/{segment_result["total_zones"]}')
    print(f'  Top drifted zones:')
    for z in segment_result['top_drifted_zones'][:5]:
        print(f'    Zone {z["zone"]}: KS={z["ks_statistic"]}, mean shift={z["mean_shift"]} trips')
    drift_found = True

    print()
    print('=' * 70)
    if drift_found:
        print('CONCLUSION: Drift detected. Recommend retraining.')
        sys.exit(1)
    else:
        print('CONCLUSION: No significant drift detected.')
        sys.exit(0)

if __name__ == '__main__':
    main()
