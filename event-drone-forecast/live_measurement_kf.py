"""Compatibility imports; no historical research module is loaded."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from event_drone_forecast.position.kalman import MeasurementKF,SimpleKF,predict,update,coefficient,PARAMS
