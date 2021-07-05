<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://www.intelrealsense.com/lidar-camera-l515/">
    <img src="https://avatars.githubusercontent.com/u/14095512?s=280&v=4" alt="Logo" width="150" height="150">
  </a>
  
# REALSENSE
Code for [`REALSENSE L515`](https://www.intelrealsense.com/lidar-camera-l515/)

## Table of Contents
- [Code](#code)
- [Requirements](#requirements)

## Code
- **merge_PLY.py**: merge 4 ply files
- **read_bag.py**: read bag file
  ```sh
  python read_bag.py -i "../BAG/<bag file>"
  ```
- **viewer_3D.py**: 3D viewer that can capture ply files ([`filter option`](https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb) can be used.)

  > **Usage**:
  ```
  [p]     Pause
  [r]     Reset View 
  [d]     Cycle through decimation values
  [z]     Toggle point scaling
  [c]     Toggle color source
  [s]     Save PNG (./out.png)
  [e]     Export points to ply (./out.ply)
  [q\ESC] Quit
  ```
### Example code
> For a list of full code examples see the [BAG_PY](./BAG_PY) folder
```python
#####################################################
##               Read bag from file                ##
#####################################################

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass
```


## Requirements
 packages in environment: 
| Name |  Version | Build  Channel  |
| :---: |  :---:  |      :---:      |
|open3d-python|             0.7.0.0                  |pypi_0    pypi|
|opencv-python|             4.2.0.34                 |pypi_0    pypi|
|pyrealsense2|              2.48.0.3381              |pypi_0    pypi|
|python|                    3.6.10               |h9f7ef89_1|

