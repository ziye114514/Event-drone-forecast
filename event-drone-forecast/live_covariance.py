"""Compatibility imports for optional continuous-time covariance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from event_drone_forecast.position.covariance import ContinuousKF,process_noise,REFERENCE_DT,SPECTRAL_DENSITY,author_Q
