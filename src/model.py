"""
src/model.py - Probabilistic HMM sequence decoding and Semi-Markov duration modeling.
"""
import numpy as np

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
N_VIDEO_FRAMES = 199319

# Match structural boundaries (video frames)
KICKOFF_1ST_HALF = 11185
END_1ST_HALF = 90651
START_2ND_HALF = 110808
KICKOFF_2ND_HALF = 113742
END_2ND_HALF = 190197

# State mapping: 0=H (Home alive), 1=A (Away alive), 2=D (Dead ball)
STATE_LABELS = {0: 'H', 1: 'A', 2: 'D'}

def viterbi_decode(emissions, transitions, initial_probs):
    """
    Standard dynamic programming Viterbi decoder for discrete hidden Markov models.
    emissions: (T, K) log-likelihood potentials
    transitions: (K, K) log-transition probabilities
    initial_probs: (K,) log initial probabilities
    """
    T, K = emissions.shape
    dp = np.zeros((T, K), dtype=np.float32)
    backtrack = np.zeros((T, K), dtype=np.int8)
    
    dp[0] = initial_probs + emissions[0]
    for t in range(1, T):
        scores = dp[t - 1, :, None] + transitions
        best_i = np.argmax(scores, axis=0)
        dp[t] = scores[best_i, np.arange(K)] + emissions[t]
        backtrack[t] = best_i
        
    path = np.zeros(T, dtype=np.int8)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        path[t] = backtrack[t + 1, path[t + 1]]
    return path

def filter_short_durations(states, min_durations):
    """
    Semi-Markov duration filtering: merges micro-segments shorter than the physical minimum.
    min_durations: dict mapping state integer to minimum duration in frames.
    """
    out = states.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        dur = j - i
        s = out[i]
        min_d = min_durations.get(s, 1)
        if dur < min_d:
            prev_s = out[i - 1] if i > 0 else (out[j] if j < n else s)
            next_s = out[j] if j < n else prev_s
            rep = prev_s if prev_s == next_s else (prev_s if i > 0 else next_s)
            out[i:j] = rep
        i = j
    return out

def build_emissions(feats, smart_h, smart_a, sp_dead, r_ctrl=2.5):
    """
    Constructs per-frame log-evidence emission potentials across states {H, A, D}.
    """
    n_frames = len(feats['ball_seen'])
    emissions = np.zeros((n_frames, 3), dtype=np.float32)
    
    b_seen = feats['ball_seen']
    dh = feats['min_d_h']
    da = feats['min_d_a']
    bx = feats['ball_x']
    by = feats['ball_y']
    b_spd = feats['ball_spd']
    p_spd = feats['mean_speed']
    
    # 1. Proximity control with velocity attenuation:
    # High-speed balls (>10 m/s) indicate flights/crosses/shots, reducing immediate controlled hold.
    ctrl_factor = np.where(np.isnan(b_spd) | (b_spd < 10.0), 1.0, np.clip(1.0 - (b_spd - 10.0) / 8.0, 0.2, 1.0))
    
    h_ctrl = np.where((dh <= r_ctrl) & (dh < da), np.clip((r_ctrl - dh) / r_ctrl, 0, 1), 0.0) * ctrl_factor
    a_ctrl = np.where((da <= r_ctrl) & (da < dh), np.clip((r_ctrl - da) / r_ctrl, 0, 1), 0.0) * ctrl_factor
    
    margin = da - dh
    margin_evidence = np.clip(margin / 3.0, -1.0, 1.0)
    
    emissions[b_seen, 0] += 2.0 * h_ctrl[b_seen] + 0.5 * margin_evidence[b_seen]
    emissions[b_seen, 1] += 2.0 * a_ctrl[b_seen] - 0.5 * margin_evidence[b_seen]
    
    # 2. Smart Tagging tactical context
    emissions[:, 0] += smart_h
    emissions[:, 1] += smart_a
    
    # 3. Dead ball: Pitch boundary excursions
    out_of_bounds = b_seen & ((bx < -0.5) | (bx > PITCH_LENGTH + 0.5) | (by < -0.5) | (by > PITCH_WIDTH + 0.5))
    emissions[out_of_bounds, 2] += 25.0
    
    # 4. Dead ball: Boundary disappearances (ball crossing lines)
    disappear = (~b_seen[1:]) & b_seen[:-1]
    disappear_idx = np.flatnonzero(disappear)
    for idx in disappear_idx:
        last_x = bx[idx]
        last_y = by[idx]
        if last_x < 1.5 or last_x > (PITCH_LENGTH - 1.5) or last_y < 1.5 or last_y > (PITCH_WIDTH - 1.5):
            end_lead = min(n_frames, idx + 100)
            emissions[idx + 1:end_lead, 2] += 2.0
            
    # 5. Dead ball: Set pieces prior
    emissions[:, 2] += sp_dead
    
    # 6. Dead ball: Low player collective movement speed (< 1.3 m/s)
    low_spd = (p_spd > 0) & (p_spd < 1.3)
    emissions[low_spd, 2] += 1.2
    
    # 7. Negative evidence against D during active play near ball
    active_ball = b_seen & (~out_of_bounds) & ((dh < 3.0) | (da < 3.0))
    emissions[active_ball, 2] -= 4.0
    
    return emissions

def predict_match_possession(feats, smart_h, smart_a, sp_dead):
    """
    Full pipeline predicting frame-accurate possession labels across the entire 199,319 frames.
    """
    n_frames = len(feats['ball_seen'])
    emissions = build_emissions(feats, smart_h, smart_a, sp_dead)
    
    # State transition log-potentials:
    # Strong turnover penalty (-15.0) enforces possession inertia.
    transitions = np.array([
        [  0.0, -15.0,  -6.5],
        [-15.0,   0.0,  -6.5],
        [ -5.0,  -5.0,   0.0]
    ], dtype=np.float32)
    
    labels = np.full(n_frames, 'X', dtype='<U1')
    
    # 2nd half pre-kickoff period is dead ball
    labels[START_2ND_HALF:KICKOFF_2ND_HALF] = 'D'
    
    # 1st Half Viterbi decoding
    init_1 = np.array([-1.0, 0.0, -10.0], dtype=np.float32) # Away kicked off
    path_1 = viterbi_decode(emissions[KICKOFF_1ST_HALF:END_1ST_HALF + 1], transitions, init_1)
    path_1 = filter_short_durations(path_1, {0: 25, 1: 25, 2: 35})
    labels[KICKOFF_1ST_HALF:END_1ST_HALF + 1] = [STATE_LABELS[s] for s in path_1]
    
    # 2nd Half Viterbi decoding
    init_2 = np.array([0.0, -1.0, -10.0], dtype=np.float32) # Home kicked off
    path_2 = viterbi_decode(emissions[KICKOFF_2ND_HALF:END_2ND_HALF + 1], transitions, init_2)
    path_2 = filter_short_durations(path_2, {0: 25, 1: 25, 2: 35})
    labels[KICKOFF_2ND_HALF:END_2ND_HALF + 1] = [STATE_LABELS[s] for s in path_2]
    
    return labels
