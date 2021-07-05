# scan_3D_090.py
# First import the library
# https://qiita.com/SatoshiGachiFujimoto/items/50d0f0a65b730647fe84
# ENV: open3d==0.9.0

import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import Open3D for easy 3d processing
import open3d as o3d

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Getting camera intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

# Streaming loop
num = 0
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color = o3d.geometry.Image(color_image)

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = (depth_image < clipping_distance) * depth_image
        depth = o3d.geometry.Image(depth_image)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Generate the pointcloud and texture mappings
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

        # Rotate
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # Estimate Normal
        pcd.estimate_normals()
        # Voxel
        voxel = pcd.voxel_down_sample(voxel_size=0.01)
        dist = np.mean(voxel.compute_nearest_neighbor_distance())
        radius = 1.5 * dist
        # Mesh
        mesh =  o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel, o3d.utility.DoubleVector([radius, radius * 2]))

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('aligned_frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('aligned_frame', images)
        key = cv2.waitKey(1)
        # Press 's' to save the point cloud
        if key & 0xFF == ord('s'):
            print("Saving to {0}.ply...".format(num))
            o3d.io.write_point_cloud('pcd-{0}.ply'.format(num), pcd)
            o3d.io.write_triangle_mesh('mesh-{0}.ply'.format(num), mesh)
            print("Done")
            num += 1
        # Press esc or 'q' to close the image window
        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()