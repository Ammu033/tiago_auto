#!/usr/bin/env python

import yaml
import subprocess
import rospy
import os
import signal
import time
import sys
from geometry_msgs.msg import PoseWithCovarianceStamped

POSE_FILE = "/home/lcastor/ros_ws/src/LCASTOR/examples/goal.yaml"
captured_pose = None

def callback(data):
    global captured_pose
    if captured_pose is None:
        captured_pose = {
            'position': {
                'x': data.pose.pose.position.x,
                'y': data.pose.pose.position.y,
                'z': data.pose.pose.position.z
            },
            'orientation': {
                'x': data.pose.pose.orientation.x,
                'y': data.pose.pose.orientation.y,
                'z': data.pose.pose.orientation.z,
                'w': data.pose.pose.orientation.w
            }
        }
        rospy.signal_shutdown("Pose captured")

def save_pose(room_name):
    rospy.init_node('amcl_pose_saver', anonymous=True)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback)

    print(f"[INFO] Waiting for AMCL pose to be published...")
    rospy.spin()

    # Load existing poses
    if os.path.exists(POSE_FILE):
        with open(POSE_FILE, 'r') as f:
            poses = yaml.safe_load(f) or {}
    else:
        poses = {}

    poses[room_name] = captured_pose

    with open(POSE_FILE, 'w') as f:
        yaml.dump(poses, f, default_flow_style=False)

    print(f"[SUCCESS] Pose for '{room_name}' saved to {POSE_FILE}")

if __name__ == "__main__":
    
    while True :
        try:
            room = input("Enter room name to save pose ('exit' to end): ").strip()
            if room =="exit":
                print("exiting")
                break
            elif room:
                save_pose(room)
            else:
                print("[ERROR] Room name is empty .")
                continue
        except rospy.ROSInterruptException:
            pass