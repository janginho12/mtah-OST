from dataclasses import dataclass
import numpy as np
import pandas as pd
from starter.control_integration.hybrid_features import _runs
from starter.detect_dead_ball import runs

@dataclass(frozen=True)
class Config:
    contact_distance_m: float=1.
    contact_duration_seconds: float=.12
    maximum_transfer_seconds: float=4.
    minimum_transfer_distance_m: float=3.
    minimum_total_distance_m: float=10.
    maximum_cycle_seconds: float=12.
    reversal_cosine_maximum: float=-.2
    own_half_fraction: float=.55
    minimum_own_half_fraction: float=.9
    minimum_observed_fraction: float=.85
    maximum_missing_seconds: float=.24
    moving_ball_mps: float=.6
    maximum_stationary_seconds: float=1.
    minimum_boundary_support_seconds: float=.3


def raw_window_ok(a,b,observed,speed,fps,cfg):
    good=observed[a:b]
    return (len(good)>0 and good.mean()>=cfg.minimum_observed_fraction
        and max((y-x for x,y in runs(~good)),default=0)<=cfg.maximum_missing_seconds*fps
        and max((y-x for x,y in runs(good&(speed[a:b]<cfg.moving_ball_mps))),default=0)
            <cfg.maximum_stationary_seconds*fps)


def circulation_cycles(contacts,ball,observed,speed,confirmed,blocks,directions,fps,pitch,cfg):
    """Only chronological adjacent control contacts may form pass chains."""
    transfers=[]
    for left,right in zip(contacts[:-1],contacts[1:]):
        a,b=left['stop']-1,right['start']
        if (left['team']!=right['team'] or left['slot']==right['slot'] or b<=a
                or b-a>cfg.maximum_transfer_seconds*fps or blocks[a]!=blocks[b]):continue
        team=left['team']
        vector=ball[b]-ball[a]
        distance=float(np.linalg.norm(vector))
        if distance<cfg.minimum_transfer_distance_m:continue
        if np.any((confirmed[a:b+1]>=0)&(confirmed[a:b+1]!=team)):continue
        if not raw_window_ok(a,b+1,observed,speed,fps,cfg):continue
        transfers.append(dict(a=a,b=b,team=team,source_slot=left['slot'],receiver_slot=right['slot'],
            vector=vector,distance=distance,left_start=left['start'],right_stop=right['stop']))
    cycles=[]
    for left,right in zip(transfers[:-1],transfers[1:]):
        a,b=left['a'],right['b']+1
        if (left['team']!=right['team'] or left['receiver_slot']!=right['source_slot']
                or right['a']<left['b'] or right['left_start']>left['right_stop']
                or b-a>cfg.maximum_cycle_seconds*fps or blocks[a]!=blocks[b-1]):continue
        cosine=float(left['vector']@right['vector']/(left['distance']*right['distance']))
        if cosine>cfg.reversal_cosine_maximum:continue
        total=left['distance']+right['distance']
        if total<cfg.minimum_total_distance_m:continue
        team=left['team']
        if np.any((confirmed[a:b]>=0)&(confirmed[a:b]!=team)):continue
        if not raw_window_ok(a,b,observed,speed,fps,cfg):continue
        direction=directions[team]*(1 if blocks[a]==1 else -1)
        own_coordinate=ball[a:b,0]/pitch[0] if direction>0 else 1-ball[a:b,0]/pitch[0]
        own_fraction=float((own_coordinate[observed[a:b]]<=cfg.own_half_fraction).mean())
        if own_fraction<cfg.minimum_own_half_fraction:continue
        cycles.append(dict(start_frame=a,end_frame_exclusive=b,team=team,half=int(blocks[a]),
            source_slot=left['source_slot']+1,middle_slot=left['receiver_slot']+1,
            last_slot=right['receiver_slot']+1,pass_distance_total_m=total,
            reversal_cosine=cosine,own_half_fraction=own_fraction))
    return cycles,transfers


def refine(original,line,setpiece,blocks,cycles,fps,cfg):
    revised=original.copy();hold=np.full(len(original),-1,np.int8);audit=[]
    # Merge overlapping witnesses only when they support the same team.
    support=np.full(len(original),-1,np.int8)
    conflict=np.zeros(len(original),bool)
    for c in cycles:
        a,b,t=c['start_frame'],c['end_frame_exclusive'],c['team']
        conflict[a:b]|=(support[a:b]>=0)&(support[a:b]!=t)
        support[a:b]=t
    support[conflict]=-1
    for index,(a,b) in enumerate(runs(original),1):
        row=dict(interval_id=index,half=int(blocks[a]),start_frame=a,end_frame_exclusive=b,
            removed_frames=0,status='no_circulation_across_D_onset',team='X')
        audit.append(row)
        if line[a:b].any():row['status']='protected_lineout';continue
        if setpiece[a:b].any():row['status']='protected_setpiece';continue
        before=round(cfg.minimum_boundary_support_seconds*fps)
        if a<before or blocks[a-before]!=blocks[a]:continue
        for lo,hi in _runs(support):
            if support[lo]<0 or not (lo<=a-before and hi>=a+before):continue
            stop=min(b,hi);team=int(support[lo])
            revised[a:stop]=False;hold[a:stop]=team
            row.update(removed_frames=stop-a,status='buildup_continuation',team='HA'[team],
                witness_start_frame=lo,witness_end_frame_exclusive=hi,revised_D_start=stop if stop<b else None)
            break
    assert not (revised&~original).any()
    np.testing.assert_array_equal(revised[line|setpiece],original[line|setpiece])
    return revised,hold,pd.DataFrame(audit),support
