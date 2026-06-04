import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
from metric_template import MetricComputer

BASELINE_PATH = '../data/demand_enriched_baseline.parquet'
NEW_DATA_PATH = '../data/demand_enriched_week4.parquet'
FEB_START = pd.Timestamp('2026-02-02')
FEB_END = pd.Timestamp('2026-02-28')

THRESHOLDS = {
    'accuracy': 0.85,
    'ks_p_value': 0.05,
    'psi': 0.10,
    'null_rate': 0.01,
    'duplicate_rate': 0.005,
}

def main():
    print('=' * 60)
    print('COMPUTE METRICS')
    print('=' * 60)

    baseline = pd.read_parquet(BASELINE_PATH)
    week4 = pd.read_parquet(NEW_DATA_PATH)
    new_data = week4[(week4['time_bucket'] >= FEB_START) & (week4['time_bucket'] <= FEB_END)]

    print(f'Baseline rows: {len(baseline)}')
    print(f'New data rows: {len(new_data)}')

    computer = MetricComputer(baseline)
    metrics = computer.compute_all_metrics(new_data)

    alerts = []

    print()
    print('=== Metric 1: Accuracy (demand shift proxy) ===')
    acc = metrics['accuracy']
    print(f'  Value: {acc:.4f}')
    if acc < THRESHOLDS['accuracy']:
        alerts.append('ALERT: Accuracy below threshold')
        print(f'  STATUS: ALERT (threshold: {THRESHOLDS["accuracy"]})')
    else:
        print(f'  STATUS: OK')

    print()
    print('=== Metric 3: Null Rates ===')
    for col, rate in metrics['null_rates'].items():
        status = 'ALERT' if rate > THRESHOLDS['null_rate'] else 'OK'
        print(f'  {col}: {rate:.4f} [{status}]')
        if status == 'ALERT':
            alerts.append(f'ALERT: High null rate in {col}')

    print()
    print('=== Metric 4: KS Test (trip_count) ===')
    ks = metrics['ks_test']
    print(f'  KS statistic: {ks["statistic"]}')
    print(f'  p-value: {ks["p_value"]}')
    if ks['drift_detected']:
        alerts.append('ALERT: KS test detected distribution drift in trip_count')
        print(f'  STATUS: DRIFT DETECTED')
    else:
        print(f'  STATUS: OK')

    print()
    print('=== Metric 5: PSI (trip_count) ===')
    psi = metrics['psi']
    print(f'  PSI: {psi}')
    if psi > THRESHOLDS['psi']:
        alerts.append('ALERT: PSI exceeds threshold')
        print(f'  STATUS: ALERT (threshold: {THRESHOLDS["psi"]})')
    else:
        print(f'  STATUS: OK (PSI < {THRESHOLDS["psi"]})')

    print()
    print('=== Metric 7: Data Freshness ===')
    freshness = metrics['data_freshness']
    print(f'  Latest record: {freshness["latest_record"]}')
    print(f'  Age: {freshness["age_hours"]} hours')
    if freshness['stale']:
        alerts.append('ALERT: Data is stale (>2 hours old)')
        print(f'  STATUS: STALE')
    else:
        print(f'  STATUS: FRESH')

    print()
    print('=== Metric 8: Duplicate Rate ===')
    dupes = metrics['duplicate_rate']
    print(f'  Duplicates: {dupes["count"]} ({dupes["rate"]:.4f})')
    if dupes['alert']:
        alerts.append('ALERT: High duplicate rate')
        print(f'  STATUS: ALERT')
    else:
        print(f'  STATUS: OK')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'baseline_rows': len(baseline),
        'new_data_rows': len(new_data),
        'metrics': {k: v for k, v in metrics.items() if k != 'accuracy_by_zone'},
        'alerts': alerts
    }
    with open(f'metrics-{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nMetrics saved to metrics-{timestamp}.json')

    print()
    print('=== SUMMARY ===')
    if alerts:
        print(f'  {len(alerts)} alert(s) fired:')
        for a in alerts:
            print(f'  - {a}')
        sys.exit(1)
    else:
        print('  All metrics within thresholds.')
        sys.exit(0)

if __name__ == '__main__':
    main()
