"""Compute the exact numbers cited in the report directly from the data."""
import os, glob, json
import pandas as pd

DATA = '/root/AI/project_root/data'
SKUBAL_ID = 669373

frames = []
for fp in sorted(glob.glob(os.path.join(DATA, 'statcast_*.parquet'))):
    df = pd.read_parquet(fp, columns=['game_pk','game_date','game_year','at_bat_number','pitch_number',
                                       'pitcher','pitch_type','release_speed','release_spin_rate',
                                       'balls','strikes','stand'])
    sk = df[df['pitcher'] == SKUBAL_ID]
    if len(sk):
        frames.append(sk)
sk = pd.concat(frames, ignore_index=True).sort_values(['game_pk','at_bat_number','pitch_number']).reset_index(drop=True)

print('TOTAL', len(sk))
print('GAMES', sk['game_pk'].nunique())
print('SEASON_ROWS')
print(sk.groupby('game_year').size().to_dict())
print('SEASON_GAMES')
print(sk.groupby('game_year')['game_pk'].nunique().to_dict())
print('SEASON_SPEED')
print(sk.groupby('game_year')['release_speed'].mean().round(2).to_dict())
print('SEASON_SPIN')
print(sk.groupby('game_year')['release_spin_rate'].mean().round(0).to_dict())

mix = sk['pitch_type'].fillna('NA').value_counts()
print('OVERALL_MIX_COUNT', mix.to_dict())
print('OVERALL_MIX_PCT', (mix/mix.sum()*100).round(2).to_dict())

# next_pitch baseline
sk['next_pitch_type'] = sk.groupby(['game_pk','at_bat_number'])['pitch_type'].shift(-1)
wt = sk.dropna(subset=['next_pitch_type'])
print('TARGET_ROWS', len(wt), 'OF', len(sk))
maj = wt['next_pitch_type'].value_counts().idxmax()
print('MAJORITY_CLASS', maj, 'ACC', round((wt['next_pitch_type']==maj).mean(), 3))

# pitch mix by season pct
season_mix = sk.assign(pitch_type=sk['pitch_type'].fillna('NA')).groupby(['game_year','pitch_type']).size().unstack(fill_value=0)
print('SEASON_MIX')
print(season_mix.to_string())
print('SEASON_MIX_PCT')
print(season_mix.div(season_mix.sum(axis=1), axis=0).mul(100).round(1).to_string())

# count situations
sk['count'] = sk['balls'].astype(str)+'-'+sk['strikes'].astype(str)
cm = sk.assign(pitch_type=sk['pitch_type'].fillna('NA')).groupby(['count','pitch_type']).size().unstack(fill_value=0)
print('COUNT_MIX_PCT')
print(cm.div(cm.sum(axis=1), axis=0).mul(100).round(1).to_string())

# stand split
stm = sk.assign(pitch_type=sk['pitch_type'].fillna('NA')).groupby(['stand','pitch_type']).size().unstack(fill_value=0)
print('STAND_MIX_PCT')
print(stm.div(stm.sum(axis=1), axis=0).mul(100).round(1).to_string())

# by_type speed/spin
print('BY_TYPE')
bt = sk.groupby(sk['pitch_type'].fillna('NA')).agg(
    n=('pitch_type','size'),
    speed_mean=('release_speed','mean'),
    spin_mean=('release_spin_rate','mean')
).round(1).sort_values('n', ascending=False)
print(bt.to_string())
