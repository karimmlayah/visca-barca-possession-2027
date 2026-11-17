"""
src/features.py - Fast streaming ATD feature extractor for More Than A Hack 2027.
Extracts ball coordinates, ball speed, player-to-ball distances, and collective speeds.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
N_VIDEO_FRAMES = 199319

def parse_team_slots(meta_path):
    """
    Parses ATD metadata to determine team membership for each of the 46 track slots.
    Returns array where 0 = Home (Barcelona), 1 = Away (Athletic Club).
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"Expected 2 teams in {meta_path}, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)

def extract_features_from_atd(raw_path, meta_path, n_frames=N_VIDEO_FRAMES):
    """
    Fast streaming extraction of ball-player kinematic features across all match frames.
    """
    team_of_track = parse_team_slots(meta_path)
    
    ball_seen = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_spd = np.full(n_frames, np.nan, dtype=np.float32)
    min_d_h = np.full(n_frames, 999.0, dtype=np.float32)
    min_d_a = np.full(n_frames, 999.0, dtype=np.float32)
    mean_speed = np.zeros(n_frames, dtype=np.float32)
    
    with open(raw_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.split(':')
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid >= n_frames:
                continue
                
            ball = parts[2].strip()
            has_b = (ball[0] != 'N') # 'N' from 'NaN'
            bx, by = np.nan, np.nan
            if has_b:
                c1 = ball.index(',')
                c2 = ball.index(',', c1 + 1)
                bx = float(ball[:c1]) * PITCH_LENGTH
                by = float(ball[c1+1:c2]) * PITCH_WIDTH
                ball_seen[fid] = True
                ball_x[fid] = bx
                ball_y[fid] = by
                b_spd_str = ball[c2+1:]
                if b_spd_str[0] != 'N':
                    ball_spd[fid] = float(b_spd_str)
                    
            p_speeds = []
            min_dh_sq = 1e8
            min_da_sq = 1e8
            
            for slot, group in enumerate(parts[1].split(';')):
                if group[0] == 'N':
                    continue
                c1 = group.index(',')
                c2 = group.index(',', c1 + 1)
                px = float(group[:c1]) * PITCH_LENGTH
                py = float(group[c1+1:c2]) * PITCH_WIDTH
                spd_str = group[c2+1:]
                if spd_str[0] != 'N':
                    p_speeds.append(float(spd_str))
                    
                if has_b:
                    dx = px - bx
                    dy = py - by
                    d2 = dx*dx + dy*dy
                    if team_of_track[slot] == 0:
                        if d2 < min_dh_sq:
                            min_dh_sq = d2
                    else:
                        if d2 < min_da_sq:
                            min_da_sq = d2
                            
            if p_speeds:
                mean_speed[fid] = sum(p_speeds) / len(p_speeds)
            if has_b:
                if min_dh_sq < 1e7:
                    min_d_h[fid] = np.sqrt(min_dh_sq)
                if min_da_sq < 1e7:
                    min_d_a[fid] = np.sqrt(min_da_sq)
                    
    return {
        'ball_seen': ball_seen,
        'ball_x': ball_x,
        'ball_y': ball_y,
        'ball_spd': ball_spd,
        'min_d_h': min_d_h,
        'min_d_a': min_d_a,
        'mean_speed': mean_speed
    }
