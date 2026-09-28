import pandas as pd
import numpy as np

df = pd.read_csv('results/heading_hardened/vfa02_heading_hardened_metrics.csv')

print('=== OVERALL COMPARISON: CURRENT VS HEADING-HARDENED ===')
b_med = df['baseline_drift_pct'].median()
h_med = df['hardened_drift_pct'].median()
b_mean = df['baseline_drift_pct'].mean()
h_mean = df['hardened_drift_pct'].mean()
b_p90 = np.percentile(df['baseline_drift_pct'], 90)
h_p90 = np.percentile(df['hardened_drift_pct'], 90)
b_max = df['baseline_drift_pct'].max()
h_max = df['hardened_drift_pct'].max()
b_pass = (df['baseline_drift_pct'] < 10.0).mean() * 100.0
h_pass = (df['hardened_drift_pct'] < 10.0).mean() * 100.0
along_err = df['along_track_err_m'].mean()
cross_err = df['cross_track_err_m'].mean()
h_rmse = df['heading_rmse_deg'].mean()

# Load along and cross track for baseline from forensic summary if available
forensic_csv = 'results/forensic_heading/vfa02_forensic_heading_summary.csv'
df_f = pd.read_csv(forensic_csv)
b_along = np.abs(df_f['heading_only_along_err_m']).mean()
b_cross = np.abs(df_f['heading_only_cross_err_m']).mean()
b_h_err = df_f['mean_vg_h_err_deg'].mean()

print(f'| Metric | Current | Heading-Hardened |')
print(f'| :--- | :---: | :---: |')
print(f'| Vfa02 Median Drift | {b_med:.2f}% | {h_med:.2f}% |')
print(f'| Mean Drift | {b_mean:.2f}% | {h_mean:.2f}% |')
print(f'| P90 Drift | {b_p90:.2f}% | {h_p90:.2f}% |')
print(f'| Max Drift | {b_max:.2f}% | {h_max:.2f}% |')
print(f'| Pass <10% | {b_pass:.1f}% | {h_pass:.1f}% |')
print(f'| Along-track Error | {b_along:.2f} m | {along_err:.2f} m |')
print(f'| Cross-track Error | {b_cross:.2f} m | {cross_err:.2f} m |')
print(f'| Heading RMSE | {b_h_err:.2f} deg | {h_rmse:.2f} deg |')

print('\n=== BREAKDOWN BY ROAD TYPE ===')
for r_type in ['straight', 'moderate_maneuver', 'complex_turn']:
    sub = df[df['road_type'] == r_type]
    b_sub_med = sub['baseline_drift_pct'].median()
    h_sub_med = sub['hardened_drift_pct'].median()
    b_sub_mean = sub['baseline_drift_pct'].mean()
    h_sub_mean = sub['hardened_drift_pct'].mean()
    h_sub_h_rmse = sub['heading_rmse_deg'].mean()
    h_sub_along = sub['along_track_err_m'].mean()
    h_sub_cross = sub['cross_track_err_m'].mean()
    h_sub_pos = sub['hardened_pos_err_m'].mean()
    print(f'-- {r_type.upper()} (N={len(sub)}) --')
    print(f'Baseline Median: {b_sub_med:.2f}%, Hardened Median: {h_sub_med:.2f}%')
    print(f'Baseline Mean:   {b_sub_mean:.2f}%, Hardened Mean:   {h_sub_mean:.2f}%')
    print(f'Mean Heading RMSE: {h_sub_h_rmse:.2f} deg')
    print(f'Mean Along-Track Error: {h_sub_along:.2f} m')
    print(f'Mean Cross-Track Error: {h_sub_cross:.2f} m')
    print(f'Mean Pos Error:    {h_sub_pos:.2f} m')

print('\n=== BREAKDOWN BY CORRIDOR TYPE ===')
for c_type in ['mapped', 'unmapped']:
    sub = df[df['corridor_type'] == c_type]
    b_sub_med = sub['baseline_drift_pct'].median()
    h_sub_med = sub['hardened_drift_pct'].median()
    b_sub_mean = sub['baseline_drift_pct'].mean()
    h_sub_mean = sub['hardened_drift_pct'].mean()
    h_sub_h_rmse = sub['heading_rmse_deg'].mean()
    h_sub_along = sub['along_track_err_m'].mean()
    h_sub_cross = sub['cross_track_err_m'].mean()
    h_sub_pos = sub['hardened_pos_err_m'].mean()
    print(f'-- {c_type.upper()} (N={len(sub)}) --')
    print(f'Baseline Median: {b_sub_med:.2f}%, Hardened Median: {h_sub_med:.2f}%')
    print(f'Baseline Mean:   {b_sub_mean:.2f}%, Hardened Mean:   {h_sub_mean:.2f}%')
    print(f'Mean Heading RMSE: {h_sub_h_rmse:.2f} deg')
    print(f'Mean Along-Track Error: {h_sub_along:.2f} m')
    print(f'Mean Cross-Track Error: {h_sub_cross:.2f} m')
    print(f'Mean Pos Error:    {h_sub_pos:.2f} m')
