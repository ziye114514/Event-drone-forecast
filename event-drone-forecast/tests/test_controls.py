"""Portable invariants: no RAW, GPU, private weights or SDK required."""
from pathlib import Path
import json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]


class Controls(unittest.TestCase):
    def test_reanchor_and_unmeasured_amplitude(self):
        from check_anchor_and_amplitude import main
        with tempfile.TemporaryDirectory() as d:
            main(Path(d)/'case')
            r=json.loads((Path(d)/'case/summary.json').read_text())
            self.assertTrue(r['position_unchanged'])
            self.assertTrue(r['containment_does_not_override_weak_old_box_or_unbounded_box_growth'])
            self.assertTrue(r['held_amplitude_does_not_refresh_expiry'])

    def test_partial_observability(self):
        from check_three_continuation import run
        with tempfile.TemporaryDirectory() as d:
            run(d);r=json.loads((Path(d)/'continuation_checks.json').read_text())
            self.assertTrue(r['two_and_axis_do_not_refresh'])

    def test_phase_rotation_wobble_and_flicker(self):
        from check_shared_phase_fft import run
        with tempfile.TemporaryDirectory() as d:
            run(d);r=json.loads((Path(d)/'controls.json').read_text())
            self.assertTrue(r['static_common_flicker_with_dominant_bright_sector_rejected'])

    def test_force_hold_expiry(self):
        from check_force_gap_hold import run
        with tempfile.TemporaryDirectory() as d:
            run(d);r=json.loads((Path(d)/'force_gap_controls.json').read_text())
            self.assertTrue(r['held_does_not_refresh_age_or_count_as_current_thrust'])


if __name__=='__main__':unittest.main()
