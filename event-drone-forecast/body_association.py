"""Call installed Ultralytics ByteTrack; return CURRENT raw detection boxes.

The library's filtered/coasting box is not a new measurement. We keep its
identity association, then recover the original current NMS box by result idx.
Low-confidence boxes cannot initialize a target. No tracker source is copied.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata
import numpy as np


class BodyAssociation:
    def __init__(self,frame_rate=25.):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        import ultralytics.trackers.byte_tracker as implementation
        self.BYTETracker=BYTETracker;self.Boxes=Boxes
        self.args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                                  track_buffer=max(1,round(.5*frame_rate)),match_thresh=.8,fuse_score=True)
        self.frame_rate=frame_rate;self.tracker=BYTETracker(self.args)
        self.target=None;self.last_t=None;self.target_seen=None;self.strong_seen=None;self.epoch=0
        self.provenance=dict(implementation='installed Ultralytics BYTETracker',
            version=importlib.metadata.version('ultralytics'),source_file=Path(implementation.__file__).name,
            source_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
            settings=vars(self.args),returned_bbox='original current NMS detection, not tracker prediction',
            low_confidence_initialization=False,maximum_weak_only_continuation_s=.5)

    def update(self,boxes,t):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing body-association timestamps required')
        if (self.last_t is not None and t-self.last_t>.5) or (self.strong_seen is not None and t-self.strong_seen>.5):
            self.tracker=self.BYTETracker(self.args);self.target=None;self.target_seen=None;self.strong_seen=None;self.epoch+=1
        self.last_t=t
        boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,6)
        tracks=self.tracker.update(self.Boxes(boxes,orig_shape=(720,1280)))
        current={s.track_id for s in self.tracker.tracked_stracks
                 if s.frame_id==self.tracker.frame_id and s.is_activated}
        observations=[]
        for track in tracks:
            identity=int(track[4]);idx=int(track[-1])
            if identity not in current or not 0<=idx<len(boxes):continue
            raw=boxes[idx]
            observations.append(dict(bbox=raw[:4].tolist(),confidence=float(raw[4]),class_id=int(raw[5]),
                                     body_track_id=identity,candidate_index=idx,low_confidence=float(raw[4])<.25,
                                     body_track_epoch=self.epoch,
                                     current_detection=True,coasting_as_detection=False))
        if self.target is None and observations:
            # A newly activated track is backed by the author's high-score gate.
            best=max(observations,key=lambda b:b['confidence'])
            self.target=best['body_track_id']
        chosen=[b for b in observations if b['body_track_id']==self.target]
        if chosen:
            self.target_seen=t
            if chosen[0]['confidence']>=.25:self.strong_seen=t
        return chosen,dict(target_track_id=self.target,current_associated=len(observations),
                           selected_current=len(chosen),target_last_seen=self.target_seen)
