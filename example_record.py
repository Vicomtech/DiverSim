from dataRecorder import DiverSim_Recorder
from utils import WeatherOpt, TimeOfDay, SimLevel
import numpy as np

if __name__ == "__main__":
    # Video and camera specifications
    fps                = 20 # Frames per second
    video_seconds      = 40 # length of the recording, in seconds
    # Initialize Recorder
    recorder   = DiverSim_Recorder(sh_path        = "/home/unreal/Linux",
                                    project_name   = "Aware2All",
                                    recording_path = "./Recordings", 
                                    colorsfile     = "etc/airsim_segmentation_palette.csv",
                                    )
    # Simulation Settings
    recorder.set_simulation_settings(level          = SimLevel.CROSSING,
                                    weather         = WeatherOpt.SUNSHINE, 
                                    light_condition = TimeOfDay.DAYLIGHT, 
                                    cubemap_size    = 1080
                                    )
    try:
        # Launch Simulator
        recorder.launch(RenderOffscreen = True)
        # Initialize Fisheye Cameras in OpenLABEL format - default values can be used
        fisheye_intrinsics, cam_extrinsics = recorder.get_default_cam_params()
        recorder.init_cameras(fisheye_intrinsics, cam_extrinsics)
        # Add trajectories for camera in each frame          
        cameraPoses = np.zeros((int(fps * video_seconds), 3))
        times       = np.arange(0, video_seconds / 2, 1 / fps)
        cameraPoses[:int(fps*video_seconds/2), 0] = -30 + 3 * times + 1/2 * (-0.15) * times ** 2 # Constant deacceleration in x axis
        # Record
        recorder.record_frames(fps, video_seconds, camPoses=cameraPoses)
    finally:
        recorder.save_and_stop()
