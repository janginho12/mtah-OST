import numpy as np
import pandas as pd
from starter.detect_dead_ball import runs

def close_dead_gaps(dead,blocks,fps=25,maximum_seconds=.5):
    dead=np.asarray(dead,bool);blocks=np.asarray(blocks)
    if (dead.ndim!=1 or dead.shape!=blocks.shape or not np.isfinite(fps) or fps<=0
            or not np.isfinite(maximum_seconds) or maximum_seconds<0):
        raise ValueError('Invalid mask/half/FPS/gap limit')
    if np.any(dead&(blocks<=0)):raise ValueError('D outside active halves')
    result=dead.copy();rows=[]
    for a,b in runs(~dead):
        if a==0 or b==len(dead) or (b-a)/fps>maximum_seconds:continue
        half=int(blocks[a-1])
        if half<=0 or not np.all(blocks[a-1:b+1]==half):continue
        result[a:b]=True
        rows.append(dict(half=half,start_frame=int(a),end_frame_exclusive=int(b),frames=int(b-a),
            duration_seconds=(b-a)/fps,left_D_last_frame=int(a-1),right_D_first_frame=int(b),
            reason='short_gap_between_D'))
    return result,pd.DataFrame(rows,columns=['half','start_frame','end_frame_exclusive','frames',
        'duration_seconds','left_D_last_frame','right_D_first_frame','reason'])
