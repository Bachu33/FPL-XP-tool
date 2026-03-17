import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import pulp
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="FPL xP Tool", layout="wide", page_icon="⚽")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0a0a0a; color: #f0f0f0; }
  section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
  .stTabs [data-baseweb="tab-list"] { background-color: #111; border-bottom: 1px solid #222; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background-color: #1a1a1a; color: #888; border-radius: 4px 4px 0 0; padding: 8px 16px; font-size: 13px; font-weight: 600; }
  .stTabs [aria-selected="true"] { background-color: #222; color: #fff; border-bottom: 2px solid #fff; }
  .stButton > button { background-color: #fff; color: #000; border: none; font-weight: 700; border-radius: 4px; transition: all 0.2s; }
  .stButton > button:hover { background-color: #ddd; }
  .stDataFrame { background-color: #111; border: 1px solid #222; border-radius: 6px; }
  div[data-testid="metric-container"] { background-color: #111; border: 1px solid #222; border-radius: 6px; padding: 12px 16px; }
  div[data-testid="metric-container"] label { color: #888; font-size: 12px; }
  div[data-testid="metric-container"] div { color: #fff; font-weight: 700; }
  .stExpander { background-color: #111; border: 1px solid #222; border-radius: 6px; }
  .stSelectbox > div, .stTextInput > div > div { background-color: #1a1a1a !important; border: 1px solid #333 !important; color: #fff !important; }
  h1,h2,h3 { color: #fff; font-weight: 700; }
  h4 { color: #aaa; font-weight: 600; }
  .good  { color: #00cc66; font-weight: 700; }
  .bad   { color: #ff4444; font-weight: 700; }
  .warn  { color: #f0a500; font-weight: 700; }
  .muted { color: #666; font-size: 12px; }
  div[data-testid="stProgress"] > div > div { background-color: #fff; }
  hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

HEADERS = {'User-Agent': 'Mozilla/5.0'}
API     = "https://fantasy.premierleague.com/api"
POS_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

ALL_CHIPS = {
    'wildcard': ('Wildcard',       '2× per season'),
    'bboost':   ('Bench Boost',    '1× per season'),
    'freehit':  ('Free Hit',       '1× per season'),
    '3xc':      ('Triple Captain', '1× per season'),
}

# Feature set from the original (proven accurate) — kept exactly.
# opp_gc_rolling5 is a strong defensive signal included in build_features.
FEATURE_COLS = [
    'pts_avg_3', 'pts_avg_5', 'pts_avg_10',
    'xg_avg_3',  'xg_avg_5',  'xg_avg_10',
    'xa_avg_3',  'xa_avg_5',  'xa_avg_10',
    'xgi_avg_3', 'xgi_avg_5',
    'mins_avg_3', 'mins_avg_5',
    'bonus_avg_3', 'bonus_avg_5',
    'minutes_ratio', 'is_home',
    'opp_attack_norm', 'opp_defence_norm',
    'opp_gc_rolling5',
    'price_norm', 'cs_rolling5',
]

# Difficulty multiplier applied to captain score only — not to raw xP predictions
DIFF_MULT = {1: 1.4, 2: 1.2, 3: 1.0, 4: 0.5, 5: 0.2}

# ─────────────────────────────────────────────────────────────────────────────
# FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bootstrap():
    data = requests.get(f"{API}/bootstrap-static/", headers=HEADERS).json()
    return pd.DataFrame(data['elements']), pd.DataFrame(data['teams'])

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fixtures():
    return pd.DataFrame(requests.get(f"{API}/fixtures/", headers=HEADERS).json())

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_histories(eligible_ids):
    all_history = []
    bar = st.progress(0, text="Fetching player histories...")
    for i, pid in enumerate(eligible_ids):
        r = requests.get(f"{API}/element-summary/{pid}/", headers=HEADERS)
        if r.status_code == 200:
            hist = r.json().get('history', [])
            if hist:
                df = pd.DataFrame(hist)
                df['player_id'] = pid
                all_history.append(df)
        if i % 50 == 0:
            bar.progress(i / len(eligible_ids),
                         text=f"Fetching histories... {i}/{len(eligible_ids)}")
            time.sleep(0.2)
    bar.empty()
    return pd.concat(all_history, ignore_index=True)

def fetch_chips(tid):
    try:
        r = requests.get(f"{API}/entry/{tid}/history/", headers=HEADERS)
        if r.status_code == 200:
            return r.json().get('chips', [])
        return []
    except:
        return []

def get_remaining_chips(chips_played):
    names_used = [c['name'] for c in chips_played]
    wc_used    = names_used.count('wildcard')
    remaining  = []
    if wc_used < 2:              remaining.append('wildcard')
    if 'bboost'  not in names_used: remaining.append('bboost')
    if 'freehit' not in names_used: remaining.append('freehit')
    if '3xc'     not in names_used: remaining.append('3xc')
    return remaining

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES — original logic preserved exactly, opp_gc_rolling5 included
# ─────────────────────────────────────────────────────────────────────────────
def build_features(history_df, fpl_teams):
    df = history_df.copy().sort_values(['player_id', 'round'])

    # Rolling averages — same columns as original
    for window in [3, 5, 10]:
        for col, feat in [
            ('total_points',               'pts'),
            ('expected_goals',             'xg'),
            ('expected_assists',           'xa'),
            ('expected_goal_involvements', 'xgi'),
            ('minutes',                    'mins'),
            ('bonus',                      'bonus'),
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[f'{feat}_avg_{window}'] = df.groupby('player_id')[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

    df['minutes_ratio'] = (df['mins_avg_5'] / 90).clip(0, 1)
    df['is_home']       = df['was_home'].astype(int)

    # Team strength normalisation
    ts = fpl_teams.set_index('id')[[
        'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away'
    ]]
    df['opp_attack_strength'] = df.apply(
        lambda r: ts.loc[r['opponent_team'], 'strength_attack_home']
        if not r['was_home'] and r['opponent_team'] in ts.index
        else ts.loc[r['opponent_team'], 'strength_attack_away']
        if r['opponent_team'] in ts.index else np.nan, axis=1)
    df['opp_defence_strength'] = df.apply(
        lambda r: ts.loc[r['opponent_team'], 'strength_defence_home']
        if r['was_home'] and r['opponent_team'] in ts.index
        else ts.loc[r['opponent_team'], 'strength_defence_away']
        if r['opponent_team'] in ts.index else np.nan, axis=1)
    for col, norm in [('opp_attack_strength',  'opp_attack_norm'),
                      ('opp_defence_strength', 'opp_defence_norm')]:
        mn, mx = df[col].min(), df[col].max()
        df[norm] = (df[col] - mn) / (mx - mn + 1e-9)

    # Opponent goals-conceded rolling — strong signal for attacking picks
    score_col = None
    for c in ['team_h_score', 'team_a_score']:
        if c in df.columns:
            score_col = c
            break
    if score_col:
        team_gc = df.groupby(['opponent_team', 'round'])[score_col].mean().reset_index()
        team_gc.columns = ['_tid', 'round', 'gsa']
        team_gc['opp_gc_rolling5'] = team_gc.groupby('_tid')['gsa'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df = df.merge(team_gc[['_tid', 'round', 'opp_gc_rolling5']],
                      left_on=['opponent_team', 'round'],
                      right_on=['_tid', 'round'], how='left').drop(columns='_tid')
    else:
        df['opp_gc_rolling5'] = 0.0

    df['price_norm']   = (df['value'] / 10 - 3.5) / (15 - 3.5)
    df['cs_rolling5']  = df.groupby('player_id')['clean_sheets'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['volatility_5'] = df.groupby('player_id')['total_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).std())
    df['goals_avg_5']  = df.groupby('player_id')['goals_scored'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()) if 'goals_scored' in df.columns else 0.0

    df['target'] = df['total_points']
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — position-aware (from latest) but same hyperparams as original
# ─────────────────────────────────────────────────────────────────────────────
def train_position_models(df):
    models, maes = {}, {}
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_df = df[df['position'] == pos].dropna(
            subset=FEATURE_COLS + ['target']).sort_values('round')
        if len(pos_df) < 50:
            continue
        split_gw = int(pos_df['round'].quantile(0.8))
        train    = pos_df[pos_df['round'] < split_gw]
        test     = pos_df[pos_df['round'] >= split_gw]
        # Same hyperparams as original — don't over-engineer
        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, verbosity=0
        )
        model.fit(train[FEATURE_COLS].fillna(0), train['target'],
                  eval_set=[(test[FEATURE_COLS].fillna(0), test['target'])],
                  verbose=False)
        preds       = model.predict(test[FEATURE_COLS].fillna(0))
        models[pos] = model
        maes[pos]   = round(mean_absolute_error(test['target'], preds), 3)
    return models, maes

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_fixture_for_gw(team_id, gw, future):
    home = future[(future['team_h'] == team_id) & (future['event'] == gw)].head(1)
    away = future[(future['team_a'] == team_id) & (future['event'] == gw)].head(1)
    if not home.empty:
        r = home.iloc[0]
        return {'is_home': True,  'opponent': r['team_a'], 'difficulty': int(r['team_h_difficulty'])}
    if not away.empty:
        r = away.iloc[0]
        return {'is_home': False, 'opponent': r['team_h'], 'difficulty': int(r['team_a_difficulty'])}
    return None

def predict_for_gw(player_row, fix, model, fpl_teams, all_att, all_def):
    ts   = fpl_teams.set_index('id')
    feat = player_row[FEATURE_COLS].copy()
    feat['is_home'] = int(fix['is_home'])
    opp_id = fix['opponent']
    if opp_id in ts.index:
        opp     = ts.loc[opp_id]
        raw_att = opp['strength_attack_home']  if not fix['is_home'] else opp['strength_attack_away']
        raw_def = opp['strength_defence_home'] if fix['is_home']     else opp['strength_defence_away']
        feat['opp_attack_norm']  = (raw_att - all_att.min()) / (all_att.max() - all_att.min() + 1e-9)
        feat['opp_defence_norm'] = (raw_def - all_def.min()) / (all_def.max() - all_def.min() + 1e-9)
    feat_df = pd.DataFrame([feat])[FEATURE_COLS].fillna(0)
    return max(0, round(float(model.predict(feat_df)[0]), 2))

def predict_next_gw(features_df, models, fpl_players, fpl_teams, fixtures, n_gws=3):
    future    = fixtures[fixtures['finished'] == False].copy()
    next_gw   = int(future['event'].dropna().min())
    ts        = fpl_teams.set_index('id')
    all_att   = features_df['opp_attack_strength'].dropna()
    all_def   = features_df['opp_defence_strength'].dropna()
    latest    = features_df.sort_values('round').groupby('player_id').last().reset_index()
    gws_ahead = sorted(future['event'].dropna().unique())[:n_gws]

    def get_next_5(team_id):
        home = future[future['team_h'] == team_id][['event', 'team_a', 'team_h_difficulty']].copy()
        home.columns = ['event', 'opponent', 'difficulty']; home['is_home'] = True
        away = future[future['team_a'] == team_id][['event', 'team_h', 'team_a_difficulty']].copy()
        away.columns = ['event', 'opponent', 'difficulty']; away['is_home'] = False
        all_f = pd.concat([home, away]).sort_values('event').head(5)
        out = []
        for _, r in all_f.iterrows():
            opp = ts.loc[r['opponent'], 'short_name'] if r['opponent'] in ts.index else '?'
            out.append(f"{'H' if r['is_home'] else 'A'} {opp}[{int(r['difficulty'])}]")
        return ' | '.join(out)

    rows = []
    for _, player in latest.iterrows():
        pid, pos = player['player_id'], player['position']
        if pos not in models:
            continue
        fpl_row = fpl_players[fpl_players['id'] == pid]
        if fpl_row.empty:
            continue
        fpl_row = fpl_row.iloc[0]
        team_id = int(fpl_row['team'])

        fix1 = get_fixture_for_gw(team_id, next_gw, future)
        if fix1 is None:
            continue
        xp1 = predict_for_gw(player, fix1, models[pos], fpl_teams, all_att, all_def)

        # Multi-GW average
        multi_xps = []
        for gw in gws_ahead:
            fix = get_fixture_for_gw(team_id, gw, future)
            if fix:
                multi_xps.append(predict_for_gw(player, fix, models[pos], fpl_teams, all_att, all_def))
        multi_xp_avg = round(np.mean(multi_xps), 2) if multi_xps else xp1

        vol        = player.get('volatility_5', 0)
        vol        = 0 if pd.isna(vol) else round(float(vol), 2)
        goals_avg5 = player.get('goals_avg_5',  0)
        goals_avg5 = 0 if pd.isna(goals_avg5) else float(goals_avg5)

        opp_id     = fix1['opponent']
        opp_short  = ts.loc[opp_id, 'short_name'] if opp_id in ts.index else '?'
        difficulty = fix1['difficulty']
        diff_mult  = DIFF_MULT.get(difficulty, 1.0)

        # Captain score: original formula (xP + 0.3*vol) × difficulty multiplier
        # GKP never captain; DEF only if regular scorer
        is_def_scorer = (pos == 'DEF' and goals_avg5 >= 0.15)
        if pos in ['MID', 'FWD'] or is_def_scorer:
            cap_score = round((xp1 + 0.3 * vol) * diff_mult, 2)
            cap_score = max(0, cap_score)
        else:
            cap_score = 0.0

        rows.append({
            'player_id':     pid,
            'player':        fpl_row['web_name'],
            'position':      pos,
            'team':          fpl_row.get('team_name', ''),
            'price':         fpl_row['now_cost'] / 10,
            'status':        fpl_row['status'],
            'news':          fpl_row.get('news', ''),
            'ownership':     float(fpl_row.get('selected_by_percent', 0)),
            'form':          float(fpl_row.get('form', 0)),
            'opponent':      f"{'H' if fix1['is_home'] else 'A'} {opp_short}",
            'difficulty':    difficulty,
            'xP':            xp1,
            'xP_multi':      multi_xp_avg,
            'volatility':    vol,
            'captain_score': cap_score,
            'next_5':        get_next_5(team_id),
            'total_points':  fpl_row.get('total_points', 0),
            'ppg':           float(fpl_row.get('points_per_game', 0)),
        })
    return pd.DataFrame(rows).sort_values('xP', ascending=False), next_gw

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def optimize_squad(predictions_df, budget=100.0, existing_squad_ids=None,
                   free_transfers=1, use_multi_gw=False):
    df = predictions_df[predictions_df['status'] == 'a'].dropna(subset=['xP']).copy()
    df['pos_int'] = df['position'].map({'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    xp_col = 'xP_multi' if use_multi_gw and 'xP_multi' in df.columns else 'xP'
    df = df.set_index('player_id')
    pids = df.index.tolist()

    prob     = pulp.LpProblem("FPL", pulp.LpMaximize)
    selected = {p: pulp.LpVariable(f"s_{p}", cat="Binary") for p in pids}
    starting = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in pids}
    captain  = {p: pulp.LpVariable(f"c_{p}", cat="Binary") for p in pids}
    vice     = {p: pulp.LpVariable(f"v_{p}", cat="Binary") for p in pids}
    if existing_squad_ids:
        tin = {p: pulp.LpVariable(f"t_{p}", cat="Binary") for p in pids}

    base  = pulp.lpSum(starting[p] * df.loc[p, xp_col]         for p in pids)
    cap_b = pulp.lpSum(captain[p]  * df.loc[p, 'captain_score'] for p in pids)
    vc_b  = pulp.lpSum(vice[p]     * df.loc[p, 'captain_score'] * 0.5 for p in pids)

    if existing_squad_ids:
        n_tin   = pulp.lpSum(tin[p] for p in pids if p not in existing_squad_ids)
        penalty = 4 * (n_tin - free_transfers)
        prob   += base + cap_b + vc_b - penalty
    else:
        prob += base + cap_b + vc_b

    prob += pulp.lpSum(selected.values()) == 15
    prob += pulp.lpSum(starting.values()) == 11
    prob += pulp.lpSum(captain.values())  == 1
    prob += pulp.lpSum(vice.values())     == 1

    for p in pids:
        prob += captain[p]  <= starting[p]
        prob += vice[p]     <= starting[p]
        prob += starting[p] <= selected[p]
        prob += captain[p] + vice[p] <= 1

    # Only eligible captain candidates (captain_score > 0)
    for p in pids:
        if df.loc[p, 'captain_score'] == 0:
            prob += captain[p] == 0
            prob += vice[p]    == 0

    for pos_int, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pp = df[df['pos_int'] == pos_int].index.tolist()
        prob += pulp.lpSum(selected[p] for p in pp) == count

    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 1].index) == 1
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 2].index) >= 3
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 3].index) >= 2
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 4].index) >= 1
    prob += pulp.lpSum(selected[p] * df.loc[p, 'price'] for p in pids) <= budget

    for team in df['team'].unique():
        tp = df[df['team'] == team].index.tolist()
        prob += pulp.lpSum(selected[p] for p in tp) <= 3

    if existing_squad_ids:
        for p in pids:
            if p not in existing_squad_ids:
                prob += tin[p] >= selected[p]
            else:
                prob += tin[p] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None

    squad_ids   = [p for p in pids if selected[p].value() == 1]
    starter_ids = [p for p in pids if starting[p].value() == 1]
    cap_id      = next(p for p in pids if captain[p].value() == 1)
    vc_id       = next(p for p in pids if vice[p].value()    == 1)

    squad_df = df.loc[squad_ids].copy().reset_index()
    squad_df['is_starting']    = squad_df['player_id'].isin(starter_ids)
    squad_df['is_captain']     = squad_df['player_id'] == cap_id
    squad_df['is_vice']        = squad_df['player_id'] == vc_id
    squad_df['is_transfer_in'] = ~squad_df['player_id'].isin(existing_squad_ids) if existing_squad_ids else False
    squad_df['role'] = squad_df.apply(
        lambda r: '★ C' if r['is_captain'] else (
            'VC' if r['is_vice'] else ('START' if r['is_starting'] else 'BENCH')), axis=1)

    n_transfers = int(squad_df['is_transfer_in'].sum()) if existing_squad_ids else 0
    hits        = max(0, n_transfers - free_transfers) * 4
    return {
        'squad':        squad_df.sort_values(['is_starting', 'pos_int'], ascending=[False, True]),
        'captain':      df.loc[cap_id, 'player'],
        'vice_captain': df.loc[vc_id, 'player'],
        'total_xP':     round(pulp.value(prob.objective), 2),
        'total_cost':   round(squad_df['price'].sum(), 1),
        'n_transfers':  n_transfers,
        'hits':         hits,
    }

def optimize_lineup_from_squad(squad_pred):
    df = squad_pred.copy()
    df['pos_int'] = df['position'].map({'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    df = df.set_index('player_id')
    pids = df.index.tolist()

    prob     = pulp.LpProblem("Lineup", pulp.LpMaximize)
    starting = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in pids}
    captain  = {p: pulp.LpVariable(f"c_{p}", cat="Binary") for p in pids}
    vice     = {p: pulp.LpVariable(f"v_{p}", cat="Binary") for p in pids}

    base  = pulp.lpSum(starting[p] * df.loc[p, 'xP']           for p in pids)
    cap_b = pulp.lpSum(captain[p]  * df.loc[p, 'captain_score'] for p in pids)
    vc_b  = pulp.lpSum(vice[p]     * df.loc[p, 'captain_score'] * 0.5 for p in pids)
    prob += base + cap_b + vc_b

    prob += pulp.lpSum(starting.values()) == 11
    prob += pulp.lpSum(captain.values())  == 1
    prob += pulp.lpSum(vice.values())     == 1

    for p in pids:
        prob += captain[p] <= starting[p]
        prob += vice[p]    <= starting[p]
        prob += captain[p] + vice[p] <= 1

    for p in pids:
        if df.loc[p, 'captain_score'] == 0:
            prob += captain[p] == 0
            prob += vice[p]    == 0

    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 1].index) == 1
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 2].index) >= 3
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 3].index) >= 2
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 4].index) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None

    starter_ids = [p for p in pids if starting[p].value() == 1]
    cap_id      = next(p for p in pids if captain[p].value() == 1)
    vc_id       = next(p for p in pids if vice[p].value()    == 1)

    df = df.reset_index()
    df['is_starting'] = df['player_id'].isin(starter_ids)
    df['is_captain']  = df['player_id'] == cap_id
    df['is_vice']     = df['player_id'] == vc_id
    df['role'] = df.apply(
        lambda r: '★ C' if r['is_captain'] else (
            'VC' if r['is_vice'] else ('START' if r['is_starting'] else 'BENCH')), axis=1)
    return df.sort_values(['is_starting', 'pos_int'], ascending=[False, True])

def analyse_weaknesses(my_pred, available):
    issues     = []
    league_avg = available.groupby('position')['xP'].mean().to_dict()
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_players = my_pred[my_pred['position'] == pos].sort_values('xP')
        avg         = league_avg.get(pos, 0)
        for _, p in pos_players.iterrows():
            if p['xP'] < avg * 0.75:
                issues.append({
                    'player':   p['player'],
                    'position': pos,
                    'xP':       p['xP'],
                    'team_avg': round(avg, 2),
                    'gap':      round(avg - p['xP'], 2),
                    'status':   p['status'],
                    'news':     p['news'],
                })
    return sorted(issues, key=lambda x: x['gap'], reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:28px;letter-spacing:-0.5px;'>⚽ FPL xP Tool</h1>",
            unsafe_allow_html=True)

with st.spinner("Loading data..."):
    fpl_players, fpl_teams = fetch_bootstrap()
    fixtures = fetch_fixtures()

team_name_map = fpl_teams.set_index('id')['name'].to_dict()
fpl_players['team_name'] = fpl_players['team'].map(team_name_map)
fpl_players['position']  = fpl_players['element_type'].map(POS_MAP)
fpl_players['price']     = fpl_players['now_cost'] / 10
future  = fixtures[fixtures['finished'] == False]
next_gw = int(future['event'].dropna().min())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    team_id_input = st.text_input("FPL Team ID", placeholder="e.g. 1234567")
    bank, free_trs, team_info, my_squad_ids = None, None, None, None
    remaining_chips = []
    chips_played    = []
    budget_input    = 100.0

    if team_id_input.strip():
        try:
            tid       = int(team_id_input)
            r_info    = requests.get(f"{API}/entry/{tid}/", headers=HEADERS)
            team_info = r_info.json() if r_info.status_code == 200 else None
            r_picks   = requests.get(
                f"{API}/entry/{tid}/event/{max(1, next_gw - 1)}/picks/", headers=HEADERS)
            if r_picks.status_code == 200:
                picks_data   = r_picks.json()
                bank         = picks_data['entry_history']['bank'] / 10
                squad_value  = picks_data['entry_history']['value'] / 10
                used_last    = picks_data['entry_history']['event_transfers']
                free_trs     = 2 if used_last == 0 else 1
                my_squad_ids = pd.DataFrame(picks_data['picks'])['element'].tolist()
                budget_input = round(squad_value + bank, 1)
                st.success("✅ Team loaded")
                if team_info:
                    st.markdown(f"**{team_info.get('name', '')}**")
                    st.markdown(f"<span class='muted'>Rank</span> **{team_info.get('summary_overall_rank','?'):,}**", unsafe_allow_html=True)
                    st.markdown(f"<span class='muted'>Points</span> **{team_info.get('summary_overall_points','?')}**", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Squad Value", f"£{round(squad_value,1)}m")
                col2.metric("Bank",        f"£{round(bank,1)}m")
                col1.metric("Budget",      f"£{budget_input}m")
                col2.metric("Free Xfers",  free_trs)

                chips_played    = fetch_chips(tid)
                remaining_chips = get_remaining_chips(chips_played)
                wc_used_count   = sum(1 for c in chips_played if c['name'] == 'wildcard')

                st.markdown("**Chips**")
                for k in remaining_chips:
                    name, note = ALL_CHIPS[k]
                    if k == 'wildcard':
                        note = f"{2 - wc_used_count} remaining"
                    st.markdown(
                        f"<div style='margin:3px 0'><span class='good'>✅ {name}</span> "
                        f"<span class='muted'>({note})</span></div>",
                        unsafe_allow_html=True)
                for c in chips_played:
                    label = ALL_CHIPS.get(c['name'], (c['name'], ''))[0]
                    st.markdown(
                        f"<span class='muted'>❌ {label} used GW{c.get('event','?')}</span>",
                        unsafe_allow_html=True)

                st.session_state['remaining_chips'] = remaining_chips
                st.session_state['chips_played']    = chips_played

        except Exception as e:
            st.warning(f"Could not load team: {e}")

    free_transfers = free_trs if free_trs else st.selectbox("Free Transfers", [1, 2])
    if not team_id_input.strip():
        budget_input = st.number_input(
            "Budget (£m)", min_value=85.0, max_value=104.0, value=100.0, step=0.1)

    st.divider()
    use_multi_gw = st.checkbox("Optimise for next 3 GWs", value=False)

    if st.button("🚀 Run Model", type="primary", use_container_width=True):
        for k in ['predictions', 'result', 'maes', 'next_gw', 'pipeline_key',
                  'scenarios', 'scenario_key']:
            st.session_state.pop(k, None)
        st.session_state['model_run'] = True

if not st.session_state.get('model_run', False):
    st.markdown(f"<p style='color:#666'>Next GW: <b style='color:#fff'>GW{next_gw}</b> — Enter your Team ID and click Run Model</p>",
                unsafe_allow_html=True)
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
eligible   = tuple(fpl_players[fpl_players['minutes'] > 90]['id'].tolist())
history_df = fetch_all_histories(eligible)

meta = fpl_players[['id', 'web_name', 'element_type', 'team', 'now_cost', 'position', 'team_name']].copy()
meta.columns = ['player_id', 'web_name', 'element_type', 'team', 'now_cost', 'position', 'team_name']
for col in ['total_points', 'minutes', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'goals_scored', 'assists', 'clean_sheets', 'bonus']:
    if col in history_df.columns:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
history_df = history_df.merge(meta, on='player_id', how='left')

cache_key = f"{next_gw}_{budget_input}_{use_multi_gw}"
if st.session_state.get('pipeline_key') != cache_key:
    with st.spinner("Building features..."):
        features_df = build_features(history_df, fpl_teams)
    with st.spinner("Training models..."):
        models, maes = train_position_models(features_df)
    with st.spinner("Generating predictions..."):
        predictions, next_gw = predict_next_gw(
            features_df, models, fpl_players, fpl_teams, fixtures, n_gws=3)
    with st.spinner("Optimizing squad..."):
        result = optimize_squad(predictions, budget=budget_input,
                                existing_squad_ids=my_squad_ids,
                                free_transfers=free_transfers,
                                use_multi_gw=use_multi_gw)
    st.session_state.update({
        'predictions':  predictions,
        'result':       result,
        'maes':         maes,
        'next_gw':      next_gw,
        'pipeline_key': cache_key,
    })
else:
    predictions = st.session_state['predictions']
    result      = st.session_state['result']
    maes        = st.session_state['maes']
    next_gw     = st.session_state['next_gw']

available       = predictions[predictions['status'] == 'a'].copy()
remaining_chips = st.session_state.get('remaining_chips', [])
chips_played    = st.session_state.get('chips_played', [])

# ── Header ────────────────────────────────────────────────────────────────────
if team_info:
    st.markdown(f"<h3>GW{next_gw} — {team_info.get('name','')}</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Rank",   f"{team_info.get('summary_overall_rank','?'):,}")
    c2.metric("Total Points",   team_info.get('summary_overall_points', '?'))
    c3.metric("Budget",         f"£{budget_input}m")
    c4.metric("Free Transfers", free_transfers)
else:
    st.markdown(f"<h3>GW{next_gw} Predictions</h3>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE GKP", maes.get('GKP'))
c2.metric("MAE DEF", maes.get('DEF'))
c3.metric("MAE MID", maes.get('MID'))
c4.metric("MAE FWD", maes.get('FWD'))
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔮 Predictions", "📋 My Lineup", "🔄 Transfers",
    "🏆 Optimal Squad", "🎯 Captain Picks", "📈 Differentials",
    "🗓️ Fixtures", "🃏 Chips"
])

# ── Tab 1: Predictions ────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"#### GW{next_gw} xP Predictions")
    c1, c2, c3 = st.columns([1, 1, 2])
    pos_filter  = c1.selectbox("Position", ["ALL", "GKP", "DEF", "MID", "FWD"], key="pred_pos")
    sort_filter = c2.selectbox("Sort by",  ["xP", "xP_multi", "captain_score", "price", "ownership", "form"], key="pred_sort")
    search      = c3.text_input("Search player", "", key="pred_search")
    disp = available.copy()
    if pos_filter != "ALL": disp = disp[disp['position'] == pos_filter]
    if search: disp = disp[disp['player'].str.lower().str.contains(search.lower())]
    disp = disp.sort_values(sort_filter, ascending=False).head(100)
    st.dataframe(
        disp[['player', 'position', 'team', 'price', 'opponent', 'xP', 'xP_multi',
              'captain_score', 'volatility', 'ownership', 'form', 'ppg', 'next_5']].reset_index(drop=True),
        column_config={
            'price':         st.column_config.NumberColumn("£",            format="£%.1f"),
            'xP':            st.column_config.NumberColumn("xP (next)",    format="%.2f"),
            'xP_multi':      st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
            'captain_score': st.column_config.NumberColumn("Cap Score",    format="%.2f"),
            'volatility':    st.column_config.NumberColumn("Vol",          format="%.2f"),
            'ownership':     st.column_config.NumberColumn("Own%",         format="%.1f%%"),
            'next_5':        st.column_config.TextColumn("Next 5 Fixtures", width=300),
        },
        use_container_width=True, hide_index=True, height=600
    )

# ── Tab 2: My Lineup ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Recommended Lineup From Your Squad")
    if not my_squad_ids:
        st.info("Enter your FPL Team ID to see your recommended lineup.")
    else:
        my_pred = predictions[predictions['player_id'].isin(my_squad_ids)].copy()
        lineup  = optimize_lineup_from_squad(my_pred)
        if lineup is not None:
            starters = lineup[lineup['is_starting']]
            bench    = lineup[~lineup['is_starting']]
            cap_row  = lineup[lineup['is_captain']]
            vc_row   = lineup[lineup['is_vice']]
            cap_name = cap_row.iloc[0]['player'] if not cap_row.empty else '?'
            vc_name  = vc_row.iloc[0]['player']  if not vc_row.empty else '?'
            c1, c2, c3 = st.columns(3)
            c1.metric("Projected xP", round(starters['xP'].sum() + (cap_row.iloc[0]['xP'] if not cap_row.empty else 0), 2))
            c2.metric("Captain",      cap_name)
            c3.metric("Vice Captain", vc_name)
            st.markdown("**Starting XI**")
            st.dataframe(
                starters[['player', 'position', 'team', 'price', 'opponent', 'xP', 'role', 'next_5']],
                column_config={
                    'price':  st.column_config.NumberColumn("£",  format="£%.1f"),
                    'xP':     st.column_config.NumberColumn("xP", format="%.2f"),
                    'next_5': st.column_config.TextColumn("Next 5", width=300),
                },
                use_container_width=True, hide_index=True
            )
            st.markdown("**Bench**")
            st.dataframe(
                bench[['player', 'position', 'team', 'price', 'xP', 'role']],
                column_config={
                    'price': st.column_config.NumberColumn("£",  format="£%.1f"),
                    'xP':    st.column_config.NumberColumn("xP", format="%.2f"),
                },
                use_container_width=True, hide_index=True
            )
        else:
            st.error("Could not generate lineup.")

        st.divider()
        st.markdown("#### 🔍 Squad Weaknesses")
        issues = analyse_weaknesses(my_pred, available)
        if not issues:
            st.success("No major weaknesses detected.")
        else:
            for issue in issues:
                gap_pct = round((issue['gap'] / max(issue['team_avg'], 0.01)) * 100)
                color   = "#ff4444" if gap_pct > 40 else "#f0a500"
                st.markdown(
                    f"<div style='background:#111;border:1px solid #222;border-radius:6px;"
                    f"padding:10px 16px;margin:4px 0;border-left:3px solid {color}'>"
                    f"<b>{issue['player']}</b> ({issue['position']}) — "
                    f"xP: <b>{issue['xP']}</b> vs avg <b>{issue['team_avg']}</b> "
                    f"<span style='color:{color}'>(-{issue['gap']} | {gap_pct}% below)</span>"
                    + (f"<br><span style='color:#888;font-size:12px'>⚠️ {issue['news']}</span>"
                       if issue['news'] else "") +
                    "</div>", unsafe_allow_html=True
                )

# ── Tab 3: Transfers ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Transfer Planner")
    if not my_squad_ids:
        st.info("Enter your FPL Team ID to get personalised transfer recommendations.")
    else:
        my_pred    = predictions[predictions['player_id'].isin(my_squad_ids)].copy()
        current_xp = my_pred['xP'].sum()
        st.markdown("**Your Current Squad**")
        st.dataframe(
            my_pred.sort_values(['position', 'xP'], ascending=[True, False])
                   [['player', 'position', 'team', 'price', 'opponent', 'xP', 'xP_multi',
                     'captain_score', 'ownership', 'status', 'news']],
            column_config={
                'price':    st.column_config.NumberColumn("£",            format="£%.1f"),
                'xP':       st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                'xP_multi': st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
            },
            use_container_width=True, hide_index=True
        )
        st.metric("Current Squad xP", round(current_xp, 2))
        st.divider()

        st.markdown(f"**Transfer Scenarios** — Bank: £{round(bank or 0, 1)}m | Free: {free_transfers}")
        col1, col2, col3 = st.columns(3)
        max_transfers = col1.slider("Max transfers", 1, 5, 2, key="max_tr_slider")
        take_hit      = col2.checkbox("Include hits", value=True, key="take_hit_cb")
        multi_gw_tr   = col3.checkbox("3GW optimise", value=False, key="multi_gw_tr")

        scenario_key = f"sc_{max_transfers}_{take_hit}_{multi_gw_tr}_{budget_input}_{free_transfers}"
        if st.session_state.get('scenario_key') != scenario_key:
            scenarios, seen = [], set()
            prog = st.progress(0, text="Generating scenarios...")
            for i, n in enumerate(range(1, max_transfers + 1)):
                prog.progress((i + 1) / max_transfers, text=f"Testing {n} transfer scenario...")
                hit = max(0, n - free_transfers) * 4
                if hit > 0 and not take_hit:
                    continue
                res = optimize_squad(predictions, budget=budget_input,
                                     existing_squad_ids=my_squad_ids,
                                     free_transfers=n, use_multi_gw=multi_gw_tr)
                if res is None:
                    continue
                new_ids = res['squad']['player_id'].tolist()
                out_ids = [p for p in my_squad_ids if p not in new_ids]
                in_ids  = [p for p in new_ids       if p not in my_squad_ids]
                if not in_ids:
                    continue
                key = (frozenset(out_ids), frozenset(in_ids))
                if key in seen:
                    continue
                seen.add(key)
                actual_hit = max(0, len(in_ids) - free_transfers) * 4
                net_gain   = round(res['total_xP'] - current_xp - actual_hit, 2)
                scenarios.append({
                    'n': len(in_ids), 'hit': actual_hit,
                    'net_gain': net_gain, 'total_xP': res['total_xP'],
                    'result': res, 'out_ids': out_ids, 'in_ids': in_ids,
                })
            prog.empty()
            st.session_state['scenarios']    = scenarios
            st.session_state['scenario_key'] = scenario_key
        else:
            scenarios = st.session_state.get('scenarios', [])

        if not scenarios:
            st.warning("No beneficial transfers found within budget.")
        else:
            for s in sorted(scenarios, key=lambda x: x['net_gain'], reverse=True):
                hit_str  = f"-{s['hit']}pt hit" if s['hit'] > 0 else "Free"
                gain_str = f"+{s['net_gain']}" if s['net_gain'] > 0 else str(s['net_gain'])
                with st.expander(
                    f"{s['n']} Transfer{'s' if s['n'] > 1 else ''} | {hit_str} | Net xP: {gain_str}",
                    expanded=(s['n'] == 1)):
                    out_players = my_pred[my_pred['player_id'].isin(s['out_ids'])]
                    in_players  = s['result']['squad'][
                        s['result']['squad']['player_id'].isin(s['in_ids'])]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("🔴 **SELL**")
                        for _, r in out_players.iterrows():
                            st.markdown(
                                f"<div style='background:#1a0a0a;border:1px solid #331111;"
                                f"border-radius:4px;padding:8px 12px;margin:4px 0'>"
                                f"<b>{r['player']}</b> £{r['price']} | xP <b>{r['xP']}</b>"
                                f"</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown("🟢 **BUY**")
                        for _, r in in_players.iterrows():
                            st.markdown(
                                f"<div style='background:#0a1a0a;border:1px solid #113311;"
                                f"border-radius:4px;padding:8px 12px;margin:4px 0'>"
                                f"<b>{r['player']}</b> £{r['price']} | "
                                f"xP <b>{r['xP']}</b> | 3GW <b>{r.get('xP_multi','?')}</b> | "
                                f"{r['opponent']}"
                                f"</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("New Squad xP", s['total_xP'])
                    col2.metric("Points Hit",   f"-{s['hit']}" if s['hit'] > 0 else "None")
                    col3.metric("Net xP Gain",  gain_str)
                    if st.checkbox("Show full squad",
                                   key=f"sq_{s['n']}_{s['hit']}_{s['net_gain']}"):
                        sq = s['result']['squad']
                        st.dataframe(
                            sq[['player', 'position', 'team', 'price', 'xP', 'xP_multi', 'role', 'is_transfer_in']],
                            column_config={
                                'price':          st.column_config.NumberColumn("£",            format="£%.1f"),
                                'xP':             st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                                'xP_multi':       st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
                                'is_transfer_in': st.column_config.CheckboxColumn("New"),
                            },
                            use_container_width=True, hide_index=True
                        )

# ── Tab 4: Optimal Squad ──────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Optimal Squad")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total xP",     result['total_xP'])
        c2.metric("Cost",         f"£{result['total_cost']}m")
        c3.metric("Captain",      result['captain'])
        c4.metric("Vice Captain", result['vice_captain'])
        if result['n_transfers'] > 0:
            st.warning(f"⚠️ {result['n_transfers']} transfers needed — -{result['hits']}pt hit")
        sq = result['squad']
        st.markdown("**Starting XI**")
        st.dataframe(
            sq[sq['is_starting']][['player', 'position', 'team', 'price', 'opponent',
                                   'xP', 'xP_multi', 'role', 'is_transfer_in', 'next_5']],
            column_config={
                'price':          st.column_config.NumberColumn("£",            format="£%.1f"),
                'xP':             st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                'xP_multi':       st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
                'is_transfer_in': st.column_config.CheckboxColumn("Transfer In"),
                'next_5':         st.column_config.TextColumn("Next 5", width=300),
            },
            use_container_width=True, hide_index=True
        )
        st.markdown("**Bench**")
        st.dataframe(
            sq[~sq['is_starting']][['player', 'position', 'team', 'price', 'xP', 'is_transfer_in']],
            column_config={
                'price':          st.column_config.NumberColumn("£",  format="£%.1f"),
                'xP':             st.column_config.NumberColumn("xP", format="%.2f"),
                'is_transfer_in': st.column_config.CheckboxColumn("Transfer In"),
            },
            use_container_width=True, hide_index=True
        )
    else:
        st.error("Optimizer could not find a valid squad.")

# ── Tab 5: Captain Picks ──────────────────────────────────────────────────────
with tab5:
    st.markdown(f"#### Captain Picks — GW{next_gw}")
    st.caption("Score = (xP + 0.3 × volatility) × fixture difficulty multiplier. "
               "Diff 5 = ×0.20  |  Diff 1 = ×1.40.  GKP excluded; DEF only if regular scorer.")
    cap_df = available[available['captain_score'] > 0].sort_values(
        'captain_score', ascending=False).head(15)
    for i, (_, r) in enumerate(cap_df.iterrows()):
        medal      = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        bg_color   = "#1a1a0a" if i == 0 else "#111"
        diff_color = "#00cc66" if r['difficulty'] <= 2 else "#f0a500" if r['difficulty'] == 3 else "#ff4444"
        st.markdown(
            f"<div style='background:{bg_color};border:1px solid #222;border-radius:6px;"
            f"padding:10px 16px;margin:4px 0'>"
            f"<span style='font-size:18px'>{medal}</span> "
            f"<b>{r['player']}</b> <span style='color:#666'>— {r['team']}</span> | "
            f"xP: <b>{r['xP']}</b> | 3GW: {r.get('xP_multi','?')} | "
            f"vs {r['opponent']} <span style='color:{diff_color}'>[diff {r['difficulty']}]</span> | "
            f"Cap Score: <b>{r['captain_score']}</b>"
            f"</div>", unsafe_allow_html=True
        )

# ── Tab 6: Differentials ──────────────────────────────────────────────────────
with tab6:
    st.markdown("#### Differentials & Value Picks")
    c1, c2 = st.columns(2)
    own_threshold = c1.slider("Max ownership %", 5, 30, 15, key="diff_own")
    min_xp        = c2.slider("Min xP", 2.0, 8.0, 3.5, step=0.5, key="diff_xp")
    diffs = available[
        (available['ownership'] <= own_threshold) &
        (available['xP'] >= min_xp)
    ].sort_values('captain_score', ascending=False)
    st.caption(f"{len(diffs)} differentials found")
    for pos in ['FWD', 'MID', 'DEF', 'GKP']:
        pos_diffs = diffs[diffs['position'] == pos].head(8)
        if pos_diffs.empty:
            continue
        st.markdown(f"**{pos}**")
        st.dataframe(
            pos_diffs[['player', 'team', 'price', 'opponent', 'xP', 'xP_multi',
                       'captain_score', 'ownership', 'next_5']].reset_index(drop=True),
            column_config={
                'price':         st.column_config.NumberColumn("£",            format="£%.1f"),
                'xP':            st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                'xP_multi':      st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
                'captain_score': st.column_config.NumberColumn("Cap Score",    format="%.2f"),
                'ownership':     st.column_config.NumberColumn("Own%",         format="%.1f%%"),
                'next_5':        st.column_config.TextColumn("Next 5",         width=300),
            },
            use_container_width=True, hide_index=True
        )

# ── Tab 7: Fixtures ───────────────────────────────────────────────────────────
with tab7:
    st.markdown("#### Fixture Difficulty Planner")
    n_gws     = st.slider("GWs ahead", 3, 8, 5, key="fix_gws")
    future_n  = fixtures[fixtures['finished'] == False].copy()
    gws_ahead = sorted(future_n['event'].dropna().unique())[:n_gws]
    ts        = fpl_teams.set_index('id')
    fix_rows  = []
    for _, team in fpl_teams.iterrows():
        row        = {'Team': team['name']}
        total_diff = 0
        for gw in gws_ahead:
            home = future_n[(future_n['team_h'] == team['id']) & (future_n['event'] == gw)]
            away = future_n[(future_n['team_a'] == team['id']) & (future_n['event'] == gw)]
            if not home.empty:
                r    = home.iloc[0]
                opp  = ts.loc[r['team_a'], 'short_name'] if r['team_a'] in ts.index else '?'
                diff = int(r['team_h_difficulty'])
                row[f'GW{int(gw)}'] = f"{opp}(H)[{diff}]"
                total_diff += diff
            elif not away.empty:
                r    = away.iloc[0]
                opp  = ts.loc[r['team_h'], 'short_name'] if r['team_h'] in ts.index else '?'
                diff = int(r['team_a_difficulty'])
                row[f'GW{int(gw)}'] = f"{opp}(A)[{diff}]"
                total_diff += diff
            else:
                row[f'GW{int(gw)}'] = "BGW"
                total_diff += 3
        row['Avg Diff'] = round(total_diff / n_gws, 1)
        fix_rows.append(row)
    fix_df  = pd.DataFrame(fix_rows).sort_values('Avg Diff')
    gw_cols = [f'GW{int(g)}' for g in gws_ahead]
    def color_diff(val):
        try:
            d = int(str(val).split('[')[1].replace(']', ''))
            if d <= 2: return 'color: #00cc66; font-weight: bold'
            if d == 3: return 'color: #f0a500'
            return 'color: #ff4444'
        except: return 'color: #666'
    st.dataframe(fix_df.style.map(color_diff, subset=gw_cols),
                 use_container_width=True, hide_index=True, height=600)

# ── Tab 8: Chips ──────────────────────────────────────────────────────────────
with tab8:
    st.markdown("#### Chip Strategy")
    if not my_squad_ids:
        st.info("Enter your FPL Team ID to see chip recommendations.")
    elif not remaining_chips:
        st.success("All chips have been used this season.")
    else:
        future_n  = fixtures[fixtures['finished'] == False].copy()
        gws_list  = sorted(future_n['event'].dropna().unique())[:10]
        gw_scores = []
        for gw in gws_list:
            gw_fix = future_n[future_n['event'] == gw]
            if gw_fix.empty: continue
            avg_diff = gw_fix[['team_h_difficulty', 'team_a_difficulty']].mean().mean()
            gw_scores.append({'gw': int(gw), 'avg_diff': round(avg_diff, 2)})
        gw_df = pd.DataFrame(gw_scores).sort_values('avg_diff') if gw_scores else pd.DataFrame()

        if 'wildcard' in remaining_chips:
            st.markdown("---")
            st.markdown("### 🃏 Wildcard")
            wc_used_count = sum(1 for c in chips_played if c['name'] == 'wildcard')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{2 - wc_used_count} use(s) remaining this season**")
                st.markdown("""
                **Use when:**
                - 4+ players you want to replace
                - Major injury crisis across multiple positions
                - Run of great fixtures for teams you don't own
                - After a double GW announcement to fully restructure
                """)
                if not gw_df.empty:
                    best = gw_df.iloc[0]
                    st.markdown(f"**Best upcoming GW:** GW{int(best['gw'])} *(avg difficulty {best['avg_diff']})*")
            with c2:
                my_pred_wc = predictions[predictions['player_id'].isin(my_squad_ids)].copy()
                st.markdown("**Weakest players to replace:**")
                for _, r in my_pred_wc.sort_values('xP').head(5).iterrows():
                    st.markdown(
                        f"<div style='background:#111;border:1px solid #222;border-radius:4px;"
                        f"padding:8px 12px;margin:3px 0'>"
                        f"<b>{r['player']}</b> ({r['position']}) — "
                        f"xP: <span style='color:#ff4444'>{r['xP']}</span>"
                        f"</div>", unsafe_allow_html=True)

        if 'bboost' in remaining_chips:
            st.markdown("---")
            st.markdown("### 🚀 Bench Boost")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1 use remaining this season**")
                st.markdown("""
                **Use when:**
                - Double gameweek with 3-4 bench players who have doubles
                - Your bench has strong fixtures that week
                - Never use in a blank gameweek
                """)
                if not gw_df.empty:
                    best = gw_df.iloc[0]
                    st.markdown(f"**Best upcoming GW:** GW{int(best['gw'])} *(avg difficulty {best['avg_diff']})*")
            with c2:
                if result:
                    bench_players = result['squad'][~result['squad']['is_starting']]
                    bench_xp      = bench_players['xP'].sum()
                    color         = "#00cc66" if bench_xp >= 12 else "#ff4444"
                    st.markdown(f"**Current bench xP:** <span style='color:{color}'>{round(bench_xp,2)}</span>",
                                unsafe_allow_html=True)
                    for _, r in bench_players.iterrows():
                        xp_col = "#00cc66" if r['xP'] > 3 else "#888"
                        st.markdown(
                            f"<div style='background:#111;border:1px solid #222;border-radius:4px;"
                            f"padding:8px 12px;margin:3px 0'>"
                            f"<b>{r['player']}</b> ({r['position']}) — "
                            f"xP: <span style='color:{xp_col}'>{r['xP']}</span> | {r['opponent']}"
                            f"</div>", unsafe_allow_html=True)
                    if bench_xp < 12:
                        st.warning("⚠️ Bench xP is low. Upgrade bench before using Bench Boost.")

        if 'freehit' in remaining_chips:
            st.markdown("---")
            st.markdown("### 🎯 Free Hit")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1 use remaining this season**")
                st.markdown("""
                **Use when:**
                - Blank gameweek — field a full 11 playing players
                - Squad resets after, never waste on a normal GW
                - Best saved for the largest blank GW of the season
                """)
                if not gw_df.empty:
                    best = gw_df.iloc[0]
                    st.markdown(f"**Best upcoming GW:** GW{int(best['gw'])} *(avg difficulty {best['avg_diff']})*")
            with c2:
                st.markdown(f"**Optimal Free Hit squad GW{next_gw}:**")
                fh_result = optimize_squad(predictions, budget=104.0,
                                           existing_squad_ids=None, free_transfers=15)
                if fh_result:
                    sq = fh_result['squad']
                    c3, c4, c5 = st.columns(3)
                    c3.metric("Total xP", fh_result['total_xP'])
                    c4.metric("Cost",     f"£{fh_result['total_cost']}m")
                    c5.metric("Captain",  fh_result['captain'])
                    st.dataframe(
                        sq[sq['is_starting']][['player', 'position', 'team', 'price', 'opponent', 'xP', 'role']],
                        column_config={
                            'price': st.column_config.NumberColumn("£",  format="£%.1f"),
                            'xP':    st.column_config.NumberColumn("xP", format="%.2f"),
                        },
                        use_container_width=True, hide_index=True
                    )

        if '3xc' in remaining_chips:
            st.markdown("---")
            st.markdown("### ⚡ Triple Captain")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1 use remaining this season**")
                st.markdown("""
                **Use when:**
                - Your captain pick has a double gameweek
                - Exceptional form + easy fixture (diff 1-2)
                - Never use on a difficulty 4-5 fixture
                """)
            with c2:
                st.markdown(f"**Best Triple Captain candidates GW{next_gw}:**")
                tc_df = available[available['captain_score'] > 0].sort_values(
                    'captain_score', ascending=False).head(5)
                for i, (_, r) in enumerate(tc_df.iterrows()):
                    diff_color = "#00cc66" if r['difficulty'] <= 2 else "#f0a500" if r['difficulty'] == 3 else "#ff4444"
                    rank_color = "#f0a500" if i == 0 else "#888"
                    st.markdown(
                        f"<div style='background:#111;border:1px solid #222;border-radius:4px;"
                        f"padding:8px 12px;margin:3px 0'>"
                        f"<span style='color:{rank_color};font-weight:700'>{i+1}.</span> "
                        f"<b>{r['player']}</b> ({r['team']}) — "
                        f"xP: <b>{r['xP']}</b> | Cap Score: <b>{r['captain_score']}</b> | "
                        f"vs {r['opponent']} <span style='color:{diff_color}'>[diff {r['difficulty']}]</span>"
                        f"</div>", unsafe_allow_html=True)