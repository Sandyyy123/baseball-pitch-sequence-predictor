"""Per-season Skubal summary across all parquet files. Output JSON to stdout."""
import os, json, glob
import pandas as pd

DATA = '/root/AI/liora_projects/03_skubal_pitch/data'
SKUBAL_ID = 669373

files = sorted(glob.glob(os.path.join(DATA, 'statcast_*.parquet')))
per_season = {}
overall_pitch = {}
schema_cols = None

for fp in files:
    season = int(os.path.basename(fp).split('_')[1].split('.')[0])
    cols_needed = ['pitcher', 'player_name', 'pitch_type', 'pitch_name',
                   'release_speed', 'release_spin_rate', 'balls', 'strikes',
                   'game_year', 'game_date']
    df = pd.read_parquet(fp, columns=[c for c in cols_needed if c is not None])
    if schema_cols is None:
        schema_cols = df.columns.tolist()
    sub = df[df['pitcher'] == SKUBAL_ID]
    if len(sub) == 0:
        per_season[season] = {'rows': 0}
        continue
    pt = sub['pitch_type'].fillna('NA').value_counts().to_dict()
    rs = sub['release_speed'].dropna()
    sr = sub['release_spin_rate'].dropna()
    per_season[season] = {
        'rows': int(len(sub)),
        'unique_games': int(sub['game_date'].nunique()) if 'game_date' in sub else None,
        'pitch_types': {str(k): int(v) for k, v in pt.items()},
        'release_speed_mean': float(rs.mean()) if len(rs) else None,
        'release_speed_std': float(rs.std()) if len(rs) else None,
        'release_speed_min': float(rs.min()) if len(rs) else None,
        'release_speed_max': float(rs.max()) if len(rs) else None,
        'spin_rate_mean': float(sr.mean()) if len(sr) else None,
        'spin_rate_std': float(sr.std()) if len(sr) else None,
        'balls_strikes_unique': int(sub.groupby(['balls','strikes']).ngroups),
    }
    for k, v in pt.items():
        overall_pitch[str(k)] = overall_pitch.get(str(k), 0) + int(v)

total_rows = sum(s.get('rows', 0) for s in per_season.values())
print(json.dumps({
    'skubal_pitcher_id': SKUBAL_ID,
    'total_rows_all_seasons': total_rows,
    'per_season': per_season,
    'overall_pitch_mix': overall_pitch,
}, indent=2, default=str))
