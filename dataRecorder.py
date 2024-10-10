import airsim
import cv2
import numpy as np
import vcd
import vcd.core as core
from vcd import scl
import time
import os
import math
from PIL import Image
import csv
import subprocess, signal
import random
import json
import utils

class DiverSim_Recorder(object):
    def __init__(self, sh_path, project_name, recording_path, colorsfile, mapillary_classes = None , numIdsPerVruClass = 1):
        '''
        Class to manage the data generation.

        Parameters
        ----------
        sh_path : str
            Path to the DiverSim executable
        project_name : str
            Name of the executable
        recording_path : str
            Path in which the simulation data will be saved
        colorsfile : str
            Path to the csv file that contains the color code for each ID in the AirSim semantic camera.
        '''
        self.sh_path = sh_path
        self.project_name = project_name
        # Recording path
        self.abs_path = os.path.join(recording_path, str(int(time.time())))
        if not(os.path.exists(self.abs_path) and os.path.isdir(self.abs_path)):
            os.makedirs(self.abs_path, exist_ok=True)
        # Segmentation path
        self.mapillary_classes = mapillary_classes
        if self.mapillary_classes is not None:
            seg_path = os.path.join(self.abs_path, "segmentation")
            self.seg_path_pinhole = os.path.join(seg_path, "pinhole")
            self.seg_path_fisheye = os.path.join(seg_path, "fisheye")
            if not(os.path.exists(self.seg_path_pinhole) and os.path.isdir(self.seg_path_pinhole)):
                os.makedirs(self.seg_path_pinhole, exist_ok=True)
            if not(os.path.exists(self.seg_path_fisheye) and os.path.isdir(self.seg_path_fisheye)):
                os.makedirs(self.seg_path_fisheye, exist_ok=True)
        # Images path
        images_path = os.path.join(self.abs_path, "images")
        self.path_pinhole = os.path.join(images_path, "pinhole")
        self.path_fisheye = os.path.join(images_path, "fisheye")
        if not(os.path.exists(self.path_pinhole) and os.path.isdir(self.path_pinhole)):
            os.makedirs(self.path_pinhole, exist_ok=True)
        if not(os.path.exists(self.path_fisheye) and os.path.isdir(self.path_fisheye)):
            os.makedirs(self.path_fisheye, exist_ok=True)
        # Initialize VCD
        self.vcd = core.OpenLABEL()
        # Taxonomy
        self.genders        = ["male", "female"]
        self.personClasses  = ["None", "white_cane", "crutches", "walking_frame", "walking_stick", "wheelchair"]
        self.edades         = ["Senior", "Adult", "Child"]
        self.races          = ["Asian", "Black", "MiddleEastern", "White"] 
        self.numIdsPerClass = numIdsPerVruClass  
        self.numVruClasses = self.numIdsPerClass * len(self.genders) * len(self.personClasses) * len(self.edades) * len(self.races)
        # Found Actors List
        self.found_actors_list  = {} 
        # # Get color list
        color_list = []
        csv_file = open(colorsfile, "r")
        csv_reader = csv.reader(csv_file, delimiter= " ")
        for num_row, row in enumerate(csv_reader):
            b = int(row[0])
            g = int(row[1])
            r = int(row[2])
            color_list.append([b,g,r])
        self.color_list = np.array(color_list)
        # Sky color
        self.skyColor = self.color_list[254, :]
        # Default values
        self.pinhole_stream_name = "Pinhole"
        self.fisheye_stream_name = "Fisheye"


    def set_simulation_settings(self, level, weather, light_condition, cubemap_size = 1080):
        '''
        Sets some simulation settings before the execution of the simulation.

        Parameters
        ----------
        level : utils.SimLevel
            Level or Map in which the simulation will take place
        weather : utils.WeatherOpt
            Weather condition in the simulation.
        light_condition : utils.TimeOfDay
            Time of day during the simulation.
        cubemap_size : int (default 1080)
            Size of each of the square pinhole cameras to be used in the cubemap. The cubemap is generated using 6 pinhole cameras
            of this resolution. The fisheye images are then generated based on this cubemap.
        '''
        # Assert values
        if not isinstance(level, utils.SimLevel):
                raise TypeError("Argument 'level' must be of type 'SimLevel'")
        if not isinstance(weather, utils.WeatherOpt):
                raise TypeError("Argument 'weather' must be of type 'weatherOpt'")
        # Assert values
        if not isinstance(light_condition, utils.TimeOfDay):
                raise TypeError("Argument 'light_condition' must be of type 'timeOfDay'")
        '''weather_opts = ["Sunshine", "Cloudy", "Overcast", "FogLow", "FogHigh", "RainLow", "RainHigh", "SnowLow", "SnowHigh"]
        daytime_opts = ["Daylight", "Twilight", "Night"]
        assert weather in weather_opts, f"Invalid weather condition: {weather}. Valid options are: {weather_opts}."
        assert light_condition in daytime_opts, f"Invalid light condition: {light_condition}. Valid options are: {daytime_opts}."'''
        # Set values
        if light_condition == utils.TimeOfDay.DAYLIGHT:
            timeOfDay = random.randint(1100, 1300)
        elif light_condition == utils.TimeOfDay.TWILIGHT:
            timeOfDay = random.randint(585, 700)
            ismorning = random.choice([True, False])
            if not ismorning:
                timeOfDay = 2400 - timeOfDay
        elif light_condition == utils.TimeOfDay.NIGHT:
            timeOfDay = random.randint(0, 500)
            isprenight = random.choice([True, False])
            if isprenight:
                timeOfDay = 2400 - timeOfDay
        # Add params to simulation settings
        sim_set_path    = os.path.join(self.sh_path, "simulation_settings.json")
        with open(sim_set_path) as f:
            sim_settings = json.load(f)
        sim_settings["map"]       = level.value
        sim_settings['weather']   = weather.value
        sim_settings['timeOfDay'] = timeOfDay
        # Set camera Position in z= 0
        sim_settings["cameraPos"]["z"] = 0
        with open(sim_set_path, 'w') as file:
            json.dump(sim_settings, file, indent=4)
        # Add context information to VCD
        uid = self.vcd.add_context(name="Weather", semantic_type="Weather")
        self.vcd.add_context_data(uid, context_data=vcd.types.text(name="value", val=weather.value))
        uid = self.vcd.add_context(name="Light Condition", semantic_type="Light Condition")
        self.vcd.add_context_data(uid, context_data=vcd.types.text(name="value", val=light_condition.value))
        # Cubemap size
        self.cubemap_size = cubemap_size
        settings_path    = os.path.join(self.sh_path, "settings.json")
        with open(settings_path, 'r') as file:
            settings_data = json.load(file)
        self.update_width_height(settings_data)
        with open(settings_path, 'w') as file:
            json.dump(settings_data, file, indent=4)
        

    def update_width_height(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ["Width", "Height"]:
                    data[key] = self.cubemap_size
                else:
                    self.update_width_height(value)
        elif isinstance(data, list):
            for item in data:
                self.update_width_height(item)

    def launch(self, RenderOffscreen = True):
        '''
        Executes the simulator
        '''
        if RenderOffscreen:
            os.system("sh %s -settings=%s -RenderOffscreen &" % (os.path.join(self.sh_path, self.project_name + ".sh"), os.path.join(self.sh_path, "settings.json")))
        else:
            os.system("sh %s -settings=%s &" % (os.path.join(self.sh_path, self.project_name + ".sh"), os.path.join(self.sh_path, "settings.json")))

    def init_cameras(self, fisheye_intrinsics, cam_extrinsics):
        '''
        Initializes the cameras to record the simulation.

         Parameters
        ----------
        fisheye_intrinsics : vcd.types.IntrinsicsFisheye
            Intrinsic parameters of the fisheye camera
        cam_extrinsics : numpy.NDArray
            Camera extrinsics with respect to the vehicle ISO8855 coordinate system. Under SCL principles, P = (R C; 0 0 0 1)
        '''
        # Initialize VCD
        self.vcd.add_stream(stream_name = self.pinhole_stream_name, uri = '', description = '', stream_type = core.StreamType.camera)
        self.vcd.add_stream(stream_name = self.fisheye_stream_name, uri = '', description = '', stream_type = core.StreamType.camera)
        # Add Odometry
        self.vcd.add_coordinate_system(name="odom", cs_type=vcd.types.CoordinateSystemType.scene_cs)
        # Vehicle-iso8855
        R = vcd.utils.euler2R([0, 0, 0])
        C = np.array([0, 0, 0]).reshape(3, 1)
        P_lcs_wrt_scene_cs = vcd.utils.create_pose(R, C)
        self.vcd.add_coordinate_system(name="vehicle-iso8855", 
                                    cs_type=vcd.types.CoordinateSystemType.local_cs,
                                    parent_name="odom",
                                    pose_wrt_parent=vcd.types.PoseData(
                                    val=list(P_lcs_wrt_scene_cs.flatten()),
                                    t_type=vcd.types.TransformDataType.matrix_4x4))
        # Add instrinsics
        self.vcd.add_stream_properties(stream_name=self.fisheye_stream_name, intrinsics=fisheye_intrinsics)
        fov = np.pi / 2
        f = self.cubemap_size / (2 * np.tan(fov/2)) 
        cx = self.cubemap_size / 2
        cy = self.cubemap_size / 2
        self.vcd.add_stream_properties(stream_name=self.pinhole_stream_name,
                                    intrinsics=vcd.types.IntrinsicsPinhole(
                                        width_px=self.cubemap_size,
                                        height_px=self.cubemap_size,
                                        camera_matrix_3x4=[f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1, 0],
                                        distortion_coeffs_1xN=None
                                    )
                                )
        # Add extrinsics
        self.vcd.add_coordinate_system(name=self.pinhole_stream_name, 
                            cs_type=vcd.types.CoordinateSystemType.sensor_cs,
                            parent_name="vehicle-iso8855", pose_wrt_parent=vcd.types.PoseData(
                                val=list(cam_extrinsics.flatten()),
                                t_type=vcd.types.TransformDataType.matrix_4x4))
        self.vcd.add_coordinate_system(name=self.fisheye_stream_name, 
                            cs_type=vcd.types.CoordinateSystemType.sensor_cs,
                            parent_name="vehicle-iso8855", pose_wrt_parent=vcd.types.PoseData(
                                val=list(cam_extrinsics.flatten()),
                                t_type=vcd.types.TransformDataType.matrix_4x4))
        # Generate Cubemap -> Fisheye maps
        self.map_x_fisheye, self.map_y_fisheye = self.get_fisheye_maps(self.cubemap_size, fisheye_intrinsics)
        # Connect AirSim client
        self.client = airsim.VehicleClient()
        # Assign IDs 
        self.assignIds()
        # Wait for segmentation to be OK
        self.waitForSegmentation()
        # Leave some margin
        time.sleep(10)
        # Set camera in specified pose
        R_unreal = utils.openlabel_to_unreal_rot(cam_extrinsics[:3, :3])
        p, y, r  = utils.unreal_rot_to_euler(R_unreal)
        cam_traslation  = airsim.Vector3r(0, 0, -cam_extrinsics[2,3]) # Can change x,y
        cam_rotation    = airsim.to_quaternion(pitch=p, roll=r, yaw=y) # Can change yaw (N * np.pi/2)
        self.client.simSetVehiclePose(airsim.Pose(cam_traslation, cam_rotation), ignore_collision=True)
        # Pause while setting Segmentations
        self.client.simPause(True)

    def record_frames(self, fps, video_seconds, annotate = True, camPoses = None):
        '''
        Generates the RGB images and Bounding Box annotations for the simulation

        Parameters
        ----------
        fps : float
            Number of frames per second.
        video_seconds : float
            Number of seconds to be recorded for this specific recording.
        annotate : Bool (Default False)
            If False, only RGB images are generated. If True, 2D BBox annotations are also generated in ASAM OpenLABEL.
        camPoses : numpy.array (Default None)
            Numpy array (numFrames x 3) that contains x, y positions and yaw for each frame. If camPoses is None, camera is still.
        '''
        # Start recording
        for framenum in range(int(fps * video_seconds)):
            print("Recording frame: %d/%d" % (framenum, int(fps * video_seconds)), end="\r")
            
            # Update Camera Position, if necessary:
            if camPoses is not None:
                current_pose = self.client.simGetVehiclePose()
                current_pos  = current_pose.position
                current_rot  = current_pose.orientation
                pitch, roll, yaw = airsim.to_eularian_angles(current_rot)
                # Update X and Y position, leave same Z
                veh_pos = airsim.Vector3r(camPoses[framenum,0], camPoses[framenum,1], current_pos.z_val)
                # Update Rotation
                veh_rot = airsim.to_quaternion(pitch=pitch, roll=roll, yaw=camPoses[framenum,2])
                self.client.simSetVehiclePose(airsim.Pose(veh_pos, veh_rot), ignore_collision=True)
            # Get Five RGB and Segmentation images
            images, segmentations = self.recordAllImages(annotate)
            # Get front pinhole and segmentation from array
            img         = images[0]
            rgb_cubemap = self.pinholes_to_cubemap(images)
            fisheye_img = cv2.remap(rgb_cubemap, self.map_x_fisheye, self.map_y_fisheye, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            if annotate:
                sgm         = segmentations[0]              
                sgm_cubemap = self.pinholes_to_cubemap(segmentations)
                fisheye_sgm = cv2.remap(sgm_cubemap, self.map_x_fisheye, self.map_y_fisheye, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            # Save images to png
            filename = f"{str(framenum)}.png"
            uri_pinhole = os.path.join(self.path_pinhole, filename)
            uri_fisheye = os.path.join(self.path_fisheye, filename)
            cv2.imwrite(uri_pinhole, img)
            cv2.imwrite(uri_fisheye, fisheye_img)
            # Add uri to VCD
            self.vcd.add_stream_properties(stream_name=self.pinhole_stream_name,
                                       stream_sync=vcd.types.StreamSync(frame_vcd=framenum),
                                       properties={"uri": uri_pinhole})
            self.vcd.add_stream_properties(stream_name=self.fisheye_stream_name,
                                       stream_sync=vcd.types.StreamSync(frame_vcd=framenum),
                                       properties={"uri": uri_fisheye})

            # Semantic Segmentation to VCD annotation
            if annotate:
                # OpenLABEL BBoxes
                self.annotateVcdBboxes(sgm,         framenum, self.pinhole_stream_name)
                self.annotateVcdBboxes(fisheye_sgm, framenum, self.fisheye_stream_name)
                # Save Mapillary format
                if self.mapillary_classes is not None:
                    mapillary_array         = self.semantic_to_mapillary(sgm)
                    mapillary_array_fisheye = self.semantic_to_mapillary(fisheye_sgm)
                    # Numpy to PIL Image format
                    mapillary_pinhole = Image.fromarray(np.uint16(mapillary_array))
                    mapillary_fisheye = Image.fromarray(np.uint16(mapillary_array_fisheye))
                    # Save mapillary semantic
                    mapillary_pinhole.save(os.path.join(self.seg_path_pinhole, str(framenum) + ".png"))
                    mapillary_fisheye.save(os.path.join(self.seg_path_fisheye, str(framenum) + ".png"))
            # Run to next frame
            self.client.simPause(False)
            time.sleep(1 / fps)
            self.client.simPause(True)


    def assignIds(self):
        # Set all IDs to 255 (white)
        success = self.client.simSetSegmentationObjectID("[\w]*", 255, is_name_regex=True)
        if not success:
            raise SystemError('There was a problem setting all segmentation object IDs to 255.')
        # Pedestrians
        all_assets = self.client.simListSceneObjects()
        self.assignIdsToPedestrians(all_assets)
        # Sky
        while True:
            found = self.client.simSetSegmentationObjectID("Ultra_Dynamic_Sky[\w]*", 254, is_name_regex=True)
            if found:
                print("sky set correctly")
                break
            else:
                print("Sky not found")
                time.sleep(1)
                break
        # Sidewalk
        self.client.simSetSegmentationObjectID("Sidewalk[\w]*", 253, is_name_regex=True)
        # Road
        self.client.simSetSegmentationObjectID("Road[\w]*", 252, is_name_regex=True)
        # Buildings
        self.client.simSetSegmentationObjectID("Building[\w]*", 251, is_name_regex=True)
        # Cars
        car_assets = [i for i in all_assets if i.startswith("Skeletal")]
        freeCarIds = 251 - (self.numVruClasses + 1)
        for car_id, car_asset in enumerate(car_assets):
            #car_id = int(car_asset[4:])
            total_id = (self.numVruClasses + 1) + car_id % freeCarIds
            self.client.simSetSegmentationObjectID(car_asset, total_id)


    def assignIdsToPedestrians(self, all_assets):
        # Go through all combinations
        # First, gender
        for g_num, gender in enumerate(self.genders):
            gender = "Pedestrian_" + gender
            g_assets =  [i for i in all_assets if i.lower().startswith(gender.lower())]
            # Edades
            for edad_num, edad in enumerate(self.edades):
                e_assets = [i for i in g_assets if i.lower().startswith((gender + "_" + edad).lower())]
                if len(g_assets) == 0:
                    continue
                # Race
                for race_num, race in enumerate(self.races):
                    r_assets = [i for i in e_assets if i.lower().startswith((gender + "_" + edad + "_" + race).lower())]
                    if len(r_assets) == 0:
                        continue
                    # Disability
                    for vru_num, vru in enumerate(self.personClasses):
                        v_assets = [i for i in r_assets if i.lower().startswith((gender + "_" + edad + "_" + race + "_" + vru).lower())]
                        for class_num, asset_name in enumerate(v_assets): 
                            classId = 0
                            # NOTE: Add +1 to distinguish from fisheye out images
                            classId += 1
                            # Add gender Id
                            classId += g_num    * self.numIdsPerClass * len(self.personClasses) * len(self.races) * len(self.edades)  
                            # Add edad
                            classId += edad_num * self.numIdsPerClass * len(self.personClasses) * len(self.races) 
                            # Add race
                            classId += race_num * self.numIdsPerClass * len(self.personClasses) 
                            # Add VRU/Disability ID
                            classId += vru_num  * self.numIdsPerClass
                            # To distinguish assets from same class
                            classId += class_num % self.numIdsPerClass
                            # Change ID to the asset
                            found = self.client.simSetSegmentationObjectID(asset_name, classId)
                            if not found:
                                print("ERROR: mesh ", asset_name, " could not be found when trying to set its segmentation ID.")
  

    def annotateVcdBboxes(self, sgm, framenum, camera_coordinate_system):
        
        # Get all different colors for the same semantic class: different instances
        reshaped_sgm = sgm.reshape((-1, 3))
        diff_colors = np.unique(reshaped_sgm, axis=0)
        for color in diff_colors:
            color_id = self.getClassIdFromColor(color)
            if color_id == 0:
                # Fisheye out black colors - ignore
                continue 
            # If color represents VRU class
            if color_id <= self.numVruClasses:
                # Correct +1 to avoid mixing with fisheye out black pixels
                color_id -= 1  
                # Check if ID in list - assume only one object per class in each scenario
                if color_id not in self.found_actors_list:
                    # Set object name
                    obj_name = "Pedestrian" + '_' + str(color_id)
                    # Add object to VCD
                    uid      = self.vcd.add_object(name=obj_name, semantic_type="Pedestrian", frame_value=framenum)
                    self.found_actors_list[color_id] = uid
                    # Get gender
                    numIdsPerGender = self.numVruClasses / len(self.genders)
                    self.setAttributeFromClassId(color_id, numIdsPerGender, "gender", uid)
                    # Get Age
                    ageId        = color_id % numIdsPerGender
                    numIdsPerAge = self.numVruClasses / len(self.genders) / len(self.edades)
                    self.setAttributeFromClassId(ageId, numIdsPerAge, "age", uid)
                    # Race
                    raceId       = ageId % numIdsPerAge
                    numIdsPerRace = self.numVruClasses / len(self.genders) / len(self.edades) / len(self.races)
                    self.setAttributeFromClassId(raceId, numIdsPerRace, "race", uid)
                    # Get VRU class
                    vruId        = raceId % numIdsPerRace
                    numIdsPerVru = self.numIdsPerClass
                    self.setAttributeFromClassId(vruId, numIdsPerVru, "object", uid)
                
                # Get uid of such item
                uid = self.found_actors_list[color_id]

                # Get mask of that color
                color_mask = cv2.inRange(sgm, color, color)
                # Bounding box of whole mask
                (x,y,bb_width,bb_height) = cv2.boundingRect(color_mask)
                bb_x_center              = int(x + bb_width/2) 
                bb_y_center              = int(y + bb_height/2)
                bbox = vcd.types.bbox(name="bbox_%s" % camera_coordinate_system, val=[bb_x_center, bb_y_center, int(bb_width), int(bb_height)], coordinate_system=camera_coordinate_system)
                # Add Data
                self.vcd.add_object_data(uid=uid, object_data=bbox, frame_value=framenum)

    def setAttributeFromClassId(self, subclassId, numIdsPerSubclass, attribute_name, uid):
        # Which attribute
        if attribute_name == "gender":
            options = self.genders
        elif attribute_name == "age":
            options = self.edades
        elif attribute_name == "race":
            options = self.races
        elif attribute_name == "object":
            options = self.personClasses
        # Get Attribute Id
        dataId  = math.floor(subclassId / numIdsPerSubclass)
        vcdData = vcd.types.text(name=attribute_name, val = options[dataId])
        self.vcd.add_object_data(uid=uid, object_data=vcdData)


    def getClassIdFromColor(self, color):
        indices = np.where(np.all(self.color_list == color, axis=1))[0]
        if len(indices) > 0:
            return indices[0]
        else:
            raise ValueError("No ID found for color %s" % str(color))


    def semantic_to_mapillary(self, sgm):
        h, w, _ = sgm.shape
        mapillary = np.zeros((h, w))
        # Get all colors present in segmentation mask
        reshaped_sgm = sgm.reshape((-1, 3))
        diff_colors = np.unique(reshaped_sgm, axis=0)
        for color in diff_colors:
            indices = np.where(np.all(sgm == color, axis=-1))
            colorId = self.getClassIdFromColor(color)
            if colorId == 0:
                # Background - set as zero
                mapillary[indices] = 0
            elif colorId <= self.numVruClasses:
                colorId -= 1
                mapillary[indices] = 256 * self.mapillary_classes["Vru"] + colorId
            elif colorId > self.numVruClasses and colorId <= 250:
                mapillary[indices] = 256 * self.mapillary_classes["Car"] + (colorId - self.numVruClasses)
            elif colorId == 251:
                mapillary[indices] = 256 * self.mapillary_classes["Building"]
            elif colorId == 252:
                mapillary[indices] = 256 * self.mapillary_classes["Road"]
            elif colorId == 253:
                mapillary[indices] = 256 * self.mapillary_classes["Pavement"]
            elif colorId == 254:
                mapillary[indices] = 256 * self.mapillary_classes["Sky"]
            elif colorId == 255:
                mapillary[indices] = 256 * self.mapillary_classes["Other"]

        return mapillary


    def get_fisheye_maps(self, cube_img_size, fisheye_intrinsics):
        # VCD initialization
        aux_vcd = core.OpenLABEL()
        aux_vcd.add_frame_properties(frame_num = 0)
        # Vehicle-iso8855
        R = vcd.utils.euler2R([0, 0, 0])
        C = np.array([0, 0, 0]).reshape(3, 1)
        aux_vcd.add_coordinate_system(name="vehicle-iso8855", 
                                    cs_type=vcd.types.CoordinateSystemType.local_cs)
                                 
        # Add cameras in VCD
        # Cubemap intrinsics
        fov = np.pi / 2
        f = cube_img_size / (2 * np.tan(fov/2)) 
        cx = cube_img_size / 2
        cy = cube_img_size / 2
        # Cubemap and other derivative camera poses
        P_scs_wrt_lcs = vcd.utils.create_pose(R, C)
        # Add cubemap stream and coordinate system 
        aux_vcd.add_coordinate_system(name="CAM_CUBEMAP", 
                                cs_type=vcd.types.CoordinateSystemType.sensor_cs,
                                parent_name="vehicle-iso8855", pose_wrt_parent=vcd.types.PoseData(
                                    val=list(P_scs_wrt_lcs.flatten()),
                                    t_type=vcd.types.TransformDataType.matrix_4x4))
        aux_vcd.add_stream(stream_name = "CAM_CUBEMAP", uri = 'uri', description = 'description', stream_type = 'stream_type')
        aux_vcd.add_stream_properties(stream_name="CAM_CUBEMAP",
                                    intrinsics=vcd.types.IntrinsicsCubemap(
                                        width_px=cube_img_size,
                                        height_px=cube_img_size,
                                        camera_matrix_3x4=[f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1, 0],
                                        distortion_coeffs_1xN=None
                                    )
                                )
        # Add fisheye stream (both aux_vcd and vcd)
        aux_vcd.add_coordinate_system(name="CAM_FISHEYE", 
                                cs_type=vcd.types.CoordinateSystemType.sensor_cs,
                                parent_name="vehicle-iso8855", pose_wrt_parent=vcd.types.PoseData(
                                    val=list(P_scs_wrt_lcs.flatten()),
                                    t_type=vcd.types.TransformDataType.matrix_4x4))
        aux_vcd.add_stream(stream_name = "CAM_FISHEYE", uri = 'uri', description = 'description', stream_type = 'stream_type')
        aux_vcd.add_stream_properties(stream_name="CAM_FISHEYE", 
                intrinsics=fisheye_intrinsics
            )    
        # Create lookup tables
        scene = scl.Scene(aux_vcd)
        map_x_fish, map_y_fish = scene.create_img_projection_maps("CAM_CUBEMAP", "CAM_FISHEYE", 0, filter_z_neg = True)
        return map_x_fish, map_y_fish
    
    def waitForSegmentation(self):
        # Get sky segmentation color value
        sky_value = self.skyColor[0]  + self.skyColor[1]*256 + self.skyColor[2] * (256**2)
        # Wait for segmentation to be set
        begin = time.time()
        while True:
            # Get segmentation image
            responses = self.client.simGetImages([
                airsim.ImageRequest("top_cubemap", airsim.ImageType.Segmentation, False, False)])
            sgm          = responses[0]
            img_rgb_1d   = np.fromstring(sgm.image_data_uint8, dtype=np.uint8) 
            img_rgb      = img_rgb_1d.reshape(sgm.height, sgm.width, 3)
            # Get center pixel in image - it should have the sky segmentation color
            center_pixel = img_rgb[int(sgm.height/2), int(sgm.width/2), :]
            center_value = center_pixel[0] + center_pixel[1]*256 + center_pixel[2]*(256**2)
            if center_value == sky_value:
                print("Segmentation finally set!")
                break
            else:
                print("Waiting for segmentation to be set, time elapsed: ", time.time() - begin)
                time.sleep(2)

    def recordAllImages(self, annotate):
        imgTypes = [airsim.ImageRequest("front_cubemap", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("left_cubemap", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("right_cubemap", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("top_cubemap", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("bottom_cubemap", airsim.ImageType.Scene, False, False)]
        if annotate:
            imgTypes.extend([airsim.ImageRequest("front_cubemap", airsim.ImageType.Segmentation, False, False),
                            airsim.ImageRequest("left_cubemap", airsim.ImageType.Segmentation, False, False),
                            airsim.ImageRequest("right_cubemap", airsim.ImageType.Segmentation, False, False),
                            airsim.ImageRequest("top_cubemap", airsim.ImageType.Segmentation, False, False),
                            airsim.ImageRequest("bottom_cubemap", airsim.ImageType.Segmentation, False, False)])
        responses = self.client.simGetImages(imgTypes)
        images        = []
        segmentations = []
        #half_responses = int(len(responses)/2)
        for r in range(5):
            img = self.to_numpy(responses[r])
            images.append(img)
            if annotate:
                sgm = self.to_numpy(responses[r + 5])
                segmentations.append(sgm)
        return images, segmentations
    
    def to_numpy(self, response):
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img   = img1d.reshape(response.height, response.width, 3)
        return img

    def pinholes_to_cubemap(self, images):
        h, w, c = images[0].shape
        cubemap = np.zeros((h * 3, w * 4, c))
        cubemap[h:h+h, w:w+w]      = images[0] #front
        cubemap[h:h+h, 0:w]        = images[1] #left
        cubemap[h:h+h, 2*w:2*w+w]  = images[2] #right
        cubemap[0:h, w:w+w]        = images[3] #top
        cubemap[2*h:2*h+h, w:w+w]  = images[4] #bottom
        if len(images) == 6:
            cubemap[h:h+h, 3*w:3*w+w] = images[5] #back
        return cubemap
    
    def get_default_cam_params(self):
        '''
        Returns default fisheye intrinsic and extrinsic values for the simulation recordings.
        '''
        fisheye_intrinsics = vcd.types.IntrinsicsFisheye(
            width_px=int(1344),
            height_px=int(968),
            lens_coeffs_1xN=[0.977512, 0.077642, 0.011963, -0.006836, 0.000613],
            center_x=660.621826, 
            center_y=485.120575,
            focal_length_x=330.928619,
            focal_length_y=330.853607,
            projection="Kannala"
        )
        # Rotation in VCD
        euler        = [-90.0, 1.27135, -89.81814]
        rot_x_rad = euler[0] * np.pi / 180.0 # pitch = -90 - rot_x
        rot_y_rad = euler[1] * np.pi / 180.0 # roll  = rot_y
        rot_z_rad = euler[2] * np.pi / 180.0 # yaw   = 90 + rot_z
        R_vcd     = vcd.utils.euler2R([rot_z_rad, rot_y_rad, rot_x_rad])
        # Translation
        translation  = [3.45613, -0.120824, 0.577209]
        T_vcd        = np.array([translation[0], translation[1], translation[2]]).reshape(3, 1)
        # Camera pose
        fisheye_extrinsics = vcd.utils.create_pose(R_vcd, T_vcd)
        return fisheye_intrinsics, fisheye_extrinsics

    def save_and_stop(self):
        '''
        Saves the generated data and stops the simulation.
        '''
        # Release videos and save VCD
        if hasattr(self, "outVideo_pinhole"):
            self.outVideo_pinhole.release()
        if hasattr(self,"outVideo_fisheye"):
            self.outVideo_fisheye.release()
        self.vcd.save(os.path.join(self.abs_path, "annotation_task1_1.json"), pretty=True)
        # Stop simulator
        self.kill_simulator_process()

    def kill_simulator_process(self):
    # Kill Simulator process
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if self.project_name in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)
                print("Process Killed!: ", str(line))


