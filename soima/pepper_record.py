# !/usr/bin/env python
# coding: utf-8


import cv2
import time
from qibullet import SimulationManager
from qibullet import PepperVirtual
import pybullet as p
import numpy as np


def main():
    # Initialize simulatore and pepper robot
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)
    pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)

    # Test Pepper with preset postures
    pepper.goToPosture("Crouch", 0.6)
    time.sleep(1)
    pepper.goToPosture("Stand", 0.6)
    time.sleep(1)
    pepper.goToPosture("StandZero", 0.6)
    time.sleep(1)

    # Define camera pointers
    handle_top = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)
    # handle_bottom = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)
    # handle_depth = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_DEPTH)
    
    # Load wall and sphere for simulation
    p.loadURDF("./data/qibullet_objs/simplebox.urdf")
    time.sleep(3)
    p.loadURDF("./data/qibullet_objs/small_sphere.urdf")
    time.sleep(3)

    # Record initial states with the option of 3 different cameras
    img_top = pepper.getCameraFrame(handle_top) 
    cv2.imshow("top camera", img_top)
#    cv2.imwrite("./data/pepper_images/top_state.png", img_top)
    cv2.waitKey(1)
#    img_bottom = pepper.getCameraFrame(handle_bottom) 
#    cv2.imshow("bottom camera", img_bottom)
#    cv2.imwrite("./data/pepper_images/bottom_state.png", img_bottom)
#    cv2.waitKey(1)
#    img_depth = pepper.getCameraFrame(handle_depth) 
#    cv2.imshow("depth camera", img_depth)
#    cv2.imwrite("./data/pepper_images/depth_state.png", img_depth)
#    cv2.waitKey(1)
    
    # Record initial joint postions
    yaw, pitch = pepper.getAnglesPosition(["HeadYaw", "HeadPitch"])
#    with open('./data/joint_positions.txt', 'a') as f:
#        f.write(str(yaw) + ' ' + str(pitch) + '\n')

    # Define range of head positions (in radians) with the sphere in view
    movement_range = np.linspace(-0.2, 0.2, 100)
    for i in range(1000):
        # Generate random positions and make movement
        x, y = np.random.choice(movement_range, size=(2, 1))
        pepper.setAngles(['HeadYaw', 'HeadPitch'], [x, y], [1, 1])
        time.sleep(1)
        
        # Record state after movement in all cameras
        img_top = pepper.getCameraFrame(handle_top) 
        cv2.imshow("top camera", img_top)
#        cv2.imwrite(
#        "./data/pepper_images/top_state" + str(i) + ".png", img_top
#        )
        cv2.waitKey(1)
#        img_bottom = pepper.getCameraFrame(handle_bottom) 
#        cv2.imshow("bottom camera", img_bottom)
#        cv2.imwrite(
#        "./data/pepper_images/bottom_state" + str(i) + ".png", img_bottom
#        )
#        cv2.waitKey(1)
#        img_depth = pepper.getCameraFrame(handle_depth) 
#        cv2.imshow("depth camera", img_depth)
#        cv2.imwrite(
#        "./data/pepper_images/depth_state" + str(i) + ".png", img_depth
#        )
#        cv2.waitKey(1)
        
        # Record joint positions after movement
        yaw, pitch = pepper.getAnglesPosition(["HeadYaw", "HeadPitch"])
#        with open('./data/joint_positions.txt', 'a') as f:
#            f.write(str(yaw) + ' ' + str(pitch) + '\n')
        print(f"Epoch: {i}")
        print(yaw, pitch)
    
    print("Done Collecting Data")
    # try:
    #     while True:
    # img = pepper.getCameraFrame(handle)
    #         cv2.imshow("bottom camera", img)
    #         cv2.waitKey(1)

    # except KeyboardInterrupt:
    #     simulation_manager.stopSimulation(client)


if __name__ == "__main__":
    main()
