from enum import Enum
import numpy as np

class WeatherOpt(Enum):
    SUNSHINE  = "Sunshine"
    CLOUDY    = "Cloudy"
    OVERCAST  = "Overcast"
    FOGLOW    = "FogLow"
    FOGHIGH   = "FogHigh"
    RAIN_LOW  = "RainLow"
    RAIN_HIGH = "RainHigh"
    SNOW_LOW  = "SnowLow"
    SNOW_HIGH = "SnowHigh"

class TimeOfDay(Enum):
    DAYLIGHT = "Daylight"
    TWILIGHT = "Twilight"
    NIGHT    = "Night"

class SimLevel(Enum):
    PARKING  = "ScenarioParking"
    CROSSING = "ScenarioCrossing"


def openlabel_to_unreal_rot(R):
    # Obtain the rotation matrix in Unreal Engine coordinate system, from OpenLABEL rotation matrix
    R_carla = np.zeros((3,3))
    R_carla[0,:] =  R[0,2],  R[0,0],  -R[0,1]
    R_carla[1,:] = -R[1,2], -R[1,0],   R[1,1]
    R_carla[2,:] =  R[2,2],  R[2,0],  -R[2,1]
    return R_carla

def unreal_rot_to_euler(R):
    if abs(R[2,0]) != 1:
        pitch  = np.arcsin(R[2, 0])
        roll   = np.arctan2( -R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch) )
        yaw    = np.arctan2( R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch) )
        pass
    else:
        yaw = 0
        if R[2,0] == 1:
            pitch = np.pi / 2
        else:
            pitch = -np.pi / 2
        roll = np.arctan2(R[1,2], R[1,1])
    return pitch, yaw, roll


