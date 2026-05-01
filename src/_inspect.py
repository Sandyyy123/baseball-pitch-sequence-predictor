import pandas as pd
import json, sys, os

base = '/root/AI/liora_projects/03_skubal_pitch/data'
df = pd.read_parquet(os.path.join(base, 'statcast_2024.parquet'))
print('SHAPE', df.shape)
print('COLS', df.columns.tolist())
# find Skubal
mask = df['player_name'].astype(str).str.contains('Skubal', case=False, na=False) if 'player_name' in df.columns else None
if mask is not None:
    sub = df[mask]
    print('SKUBAL_2024_ROWS', len(sub))
    if 'pitcher' in sub.columns and len(sub):
        print('SKUBAL_PITCHER_ID', sub['pitcher'].iloc[0])
    if 'pitch_type' in sub.columns:
        print('PITCH_TYPES_2024')
        print(sub['pitch_type'].value_counts(dropna=False).to_dict())
print('DTYPES_HEAD')
print(df.dtypes.head(30))
