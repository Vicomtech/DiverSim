# DiverSim

**DiverSim** is an innovative simulating tool to generate **synthetic pedestrian data** with a focus on **diversity** and inclusion. Built on Unreal Engine 5, DiverSim creates balanced datasets featuring equal proportions of **genders, ethnicities**, and individuals with **disabilities**. Users can customize various simulation parameters, including atmospheric conditions, **fisheye camera** parameters and characteristics of the pedestrians, while data is annotated in **ASAM OpenLABEL** format. As an open-source tool, DiverSim enables researchers and developers to train and validate AI models that effectively represent diverse pedestrian scenarios.

![DiverSim fisheye camera capture, showing pedestrians of different characteristics](./images/fisheye_capture.png "DiverSim fisheye camera capture")

## How To Use
To understand how to perform a recording with the DiverSim tool, please refer to the *example_record.py* script. This script outlines the essential simulation parameters, specifies the path to the executable (see links below), and designates the save location for the output data. It effectively manages the execution of the simulator, configures the simulation settings, and captures and annotates the generated data.

## Simulation Settings

A file called `simulation_settings.json` is placed next to the packaged executable (see download link below) that controls different variables during the simulation. This file can be modified either manually or using the Python API, as shown in the *example_record.py* script.

| Json field | Description |
| --- | --- | 
| `map` | Chooses between the different scenarios for the simulation. Options are:  <br /> - `ScenarioCrossing` or `0` <br /> - `ScenarioParking` or `1` <br /> - `ScenarioPickup` or `2` (not available yet)|  
| `accelerateBeginning` | Accelerates the speed of the simulation at the beginning for 2 seconds. This ensures slow walking classes appear crossing the street. | 
| `timeScale` | Controls the speed of the pedestrian simulation. `1` is normal speed, `0.5` is half speed, `2` is double speed.| 
| `timeOfDay` | Position of the sun/moon. Value goes from `0` to `2400` (hundreths of an hour). Car lights and street lights turn off during the day, between `[650, 1750]` which are sunrise and sunset, respectively. |
| `sunAngle` | Angle of the sun around z-axis. Range between `0` and `360`. If set to `-1`, the angle will be random on every simulation (very useful for generating diverse light conditions). |
| `weather` | Controls the appearence of the sky. Current options are: `Sunshine`, `Cloudy`, `Overcast`, `FogLow`, `FogHigh`, `RainLow`, `RainHigh`, `SnowLow`, `SnowHigh`.  |
| `maxAgentCount` | Maximum number of agents allowed to exist concurrently in the simulation. |
| `spawnPolicy` | Decides how agents are spawned. The different modes are <br /> - `random` Agents are spawned randomly regardless of class. <br /> - `DisabilityNormalized` Each disability class is given a chance to spawn, indicated by the `spawnChanceDisability` field. <br /> - `DisabilityNormalizedUnique` Same as above but all pedestrian classes are unique, no two same pedestrians with the same characteristics will spawn. This is the policy that should be selected for the ground truth 2D bounding box annotation.|
| `spawnChanceDisability` | How likely it is for a disability class to spawn. <br /> Values are normalized afterwards, so `(A = 1, B = 2)` is the same as `(A = 0.33, B = 0.66)`. |
| `carDensity` | Percentage of car slots occupied by cars (parking, road, etc). Range between `0` and `1`, and `-1`for random. |
| `cameraPosition` | 3D Vector that determines the spawn position of the camera |


## Simulation Controls

| Key | Description |
| --- | --- |
`WASD` | Pitch and Yaw of camera.
`QE` | Roll of camera.
`ArrowKeys` | Move forwards or sideways.
`Backspace` | Reset simulation. The `simulation_settings.json` file is reloaded when doing this, so it can be edited to change the simulation parameters without closing the sim.
`Escape` | Close the simulation.

## Licensing
[Crutches](https://skfb.ly/6WMZr) by Mouch is licensed under [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/).   
[Bridge street light 3](https://skfb.ly/oxpWT) by Streetlights & ETC is licensed under [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/).

All the assets and animations used in this simulator have been carefully chosen so that their licenses allow their use for AI training and validation purposes.

The licensing of DiverSim is subject to the **Unreal Engine End User License Agreement (EULA)**. Please refer to the Unreal Engine EULA for specific details regarding the usage and distribution of the simulator. The EULA sets the terms and conditions for using Unreal Engine and any content generated using it, including this tool.

As a user of DiverSim, it is your responsibility to comply with the terms specified in the Unreal Engine EULA. Ensure that you read and understand the EULA before using this simulator for any purpose.

## Atributions
DiverSim was generated employing the AirSim Simulator on Unreal Engine 5 as part of the European project AWARE2ALL.

## Citation
If you use DiverSim in your research, publications, or academic work, please cite the following paper:
```
@inproceedings{IniguezDeGordoa2025,
  author    = {Iñiguez de Gordoa, J. A. and Hormaetxea, M. and Nieto, M. and Vélez, G. and Mujika, A.},
  title     = {DiverSim: A Customizable Simulation Tool to Generate Diverse Vulnerable Road User Datasets},
  booktitle = {Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems},
  year      = {2025},
  isbn      = {978-989-758-745-0},
  issn      = {2184-495X},
  pages     = {17-24}
}
```
Or in Harvard Citation Format:
> Iñiguez de Gordoa, J. A.; Hormaetxea, M.; Nieto, M.; Vélez, G. and Mujika, A. (2025). DiverSim: A Customizable Simulation Tool to Generate Diverse Vulnerable Road User Datasets.  In _Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems_, ISBN 978-989-758-745-0, ISSN 2184-495X, pages 17-24.    

## Contact
For any questions or inquiries regarding the DiverSim simulation tool, please contact jainiguez@vicomtech.org or mhormaetxea@vicomtech.org.

## Download links
- You can download the simulator executable [here](https://opendatasets.vicomtech.org/di21-diversim/ceb2d330).
- You can download data samples generated with DiverSim [here](https://opendatasets.vicomtech.org/di21-diversim-example-data/9187bf5b).
