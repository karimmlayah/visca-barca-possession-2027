"""
src/priors.py - Tactical context and set-piece stoppage priors from Smart Tagging XML.
"""
import xml.etree.ElementTree as ET
import numpy as np

N_VIDEO_FRAMES = 199319
FPS = 25.0

ATTACKING_CODES = {
    'BUILD UP', 'PROGRESSION', 'ATTACKING TRANSITION', 
    'BALL IN FINAL THIRD', 'BALL IN THE BOX', 'CREATING CHANCES'
}

DEFENDING_CODES = {
    'DEFENDING IN DEFENSIVE THIRD', 'DEFENDING IN MIDDLE THIRD', 
    'DEFENDING IN ATTACKING THIRD', 'DEFENSIVE TRANSITION'
}

def load_smart_tagging_priors(xml_path, n_frames=N_VIDEO_FRAMES):
    """
    Parses SportsCode XML and constructs per-frame log-evidence priors for:
    - Home team attacking phases
    - Away team attacking phases
    - Set piece stoppage dead-ball windows
    """
    smart_prior_h = np.zeros(n_frames, dtype=np.float32)
    smart_prior_a = np.zeros(n_frames, dtype=np.float32)
    set_piece_dead = np.zeros(n_frames, dtype=np.float32)
    
    root = ET.parse(xml_path).getroot()
    for instance in root.iter("instance"):
        code_elem = instance.findtext("code") or ""
        code = code_elem.strip()
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        
        sf = max(0, int(round(start_s * FPS)))
        ef = min(n_frames, int(round(end_s * FPS)))
        
        tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
        team = tags.get("Team", "")
        
        if team == "Barcelona":
            if code in ATTACKING_CODES:
                smart_prior_h[sf:ef] += 0.40
            elif code in DEFENDING_CODES:
                smart_prior_a[sf:ef] += 0.40
        elif team == "Atletic Club":
            if code in ATTACKING_CODES:
                smart_prior_a[sf:ef] += 0.40
            elif code in DEFENDING_CODES:
                smart_prior_h[sf:ef] += 0.40
                
        if code.lower() == "set pieces":
            stype = (tags.get("Type") or "").lower()
            if "kick off" in stype:
                set_piece_dead[max(0, sf - 200):sf] = 3.5
            else:
                # 35s window: preparation stoppage occurs prior to the restart kick
                mid = int(sf + 0.65 * (ef - sf))
                set_piece_dead[sf:mid] = 3.5
                
    return smart_prior_h, smart_prior_a, set_piece_dead
