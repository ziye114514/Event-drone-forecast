"""Current observed direction plus explicitly held recent period amplitude.

Neither current rotor frequency nor gain-learning evidence is fabricated.
Same0.12s missing-input limit as the existing total-acceleration hold.
"""
import numpy as np
from visibility_period import force_input


def current_axis_qualified(structure):
    side=structure.get('axis_support') or {}
    return bool((side.get('valid') and side.get('confidence',0)>=.45) or
        structure.get('pose_source')=='current_qualified_visible_conics' or
        (structure.get('mode') in ('four','three','partial_two') and structure.get('continuous',{}).get('updated')))


class AxisAmplitudeBridge:
    def __init__(self):self.last=None;self.last_t=None;self.epoch=None

    def step(self,t,structure,u,quality,period):
        epoch=structure.get('geometry_epoch')
        if (self.last_t is not None and t-self.last_t>.5) or (self.epoch is not None and epoch!=self.epoch):self.last=None
        self.last_t=t;self.epoch=epoch
        relative=period.get('relative_f2')
        if u is not None and quality>=.35 and relative is not None:
            self.last=(t,float(relative))
            return u,quality,dict(applied=False,amplitude_current=True,learning_input_current=True,reason='current_complete_input')
        current_axis=current_axis_qualified(structure)
        if self.last is None or not current_axis or t-self.last[0]>.12+1e-9:
            return u,quality,dict(applied=False,amplitude_current=False,learning_input_current=True,reason='no_recent_amplitude_or_current_axis')
        revised,q=force_input(structure,self.last[1],True)
        if revised is None:return u,quality,dict(applied=False,amplitude_current=False,learning_input_current=True,reason='current_axis_unqualified')
        return revised,q,dict(applied=True,amplitude_current=False,learning_input_current=False,
            reason='current_axis_with_recent_period_amplitude',period_age_s=t-self.last[0],maximum_period_hold_s=.12,
            held_relative_f2=self.last[1],current_frequency_invented=False,hidden_rotor_frequencies_added=0,
            held_amplitude_excluded_from_gain_and_input_integral_learning=True)
