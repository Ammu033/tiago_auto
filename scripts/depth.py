#!/usr/bin/env python3

# ROS libraries
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import message_filters

# Vision libraries
import cv2
import numpy as np
from ultralytics import YOLO

# Threading for terminal input
import threading


class YoloSegNode:
    def __init__(self):
        rospy.init_node('yolo_seg_node', anonymous=True)

        # -------------------------------------------------------
        # Load YOLO segmentation model
        # -------------------------------------------------------
        rospy.loginfo("[INFO] Loading YOLO model...")
        self.model = YOLO('yolov8n-seg.pt')
        rospy.loginfo("[INFO] Model loaded!")

        self.bridge = CvBridge()

        # -------------------------------------------------------
        # State variables
        # -------------------------------------------------------
        self.table_detected   = False
        self.detected_tables  = []   # list of detected tables sorted left to right
        self.selected_table   = None # table user selected
        self.navigating       = False
        self.stopped          = False

        # -------------------------------------------------------
        # Connect to move_base
        # -------------------------------------------------------
        rospy.loginfo("[INFO] Connecting to move_base...")
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        rospy.loginfo("[INFO] Connected to move_base!")

        # -------------------------------------------------------
        # Publishers
        # -------------------------------------------------------
        self.mask_pub      = rospy.Publisher('/yolo/binary_mask',     Image, queue_size=1)
        self.annotated_pub = rospy.Publisher('/yolo/annotated_image', Image, queue_size=1)
        self.cmd_vel_pub   = rospy.Publisher('/cmd_vel',              Twist, queue_size=1)

        # -------------------------------------------------------
        # Synchronized subscribers for RGB + Depth
        # We sync them so each RGB frame has matching depth frame
        # -------------------------------------------------------
        rgb_sub   = message_filters.Subscriber('/xtion/rgb/image_raw',   Image)
        depth_sub = message_filters.Subscriber('/xtion/depth/image_raw', Image)

        # ApproximateTimeSynchronizer syncs RGB and depth by timestamp
        # slop=0.1 means 100ms tolerance between frames
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # -------------------------------------------------------
        # Control loop timer — runs every 100ms
        # -------------------------------------------------------
        rospy.Timer(rospy.Duration(0.1), self.control_loop)

        # -------------------------------------------------------
        # Start terminal input thread
        # Runs in background so it doesnt block ROS
        # -------------------------------------------------------
        self.input_thread = threading.Thread(target=self.terminal_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        rospy.loginfo("[INFO] Node started! Waiting for images...")

    # -------------------------------------------------------
    # Terminal input — runs in background thread
    # Waits for user to type table number
    # -------------------------------------------------------
    def terminal_input(self):
        while not rospy.is_shutdown():
            try:
                if self.detected_tables:
                    user_input = input("\n[INPUT] Enter table number to navigate to: ")
                    table_num  = int(user_input)

                    if 1 <= table_num <= len(self.detected_tables):
                        self.selected_table = self.detected_tables[table_num - 1]
                        rospy.loginfo(f"[INFO] Navigating to Table {table_num}!")
                        self.navigate_to_table(self.selected_table)
                    else:
                        rospy.logwarn(f"[WARN] Invalid table number! Choose between 1 and {len(self.detected_tables)}")
            except ValueError:
                rospy.logwarn("[WARN] Please enter a valid number!")

    # -------------------------------------------------------
    # Navigate to selected table using depth data
    # Stops at 0.5m from the table
    # -------------------------------------------------------
    def navigate_to_table(self, table_info):
        depth       = table_info['depth']   # depth in meters
        center_x    = table_info['cx']      # pixel x center of table
        frame_width = table_info['frame_w'] # frame width

        # -------------------------------------------------------
        # Convert pixel position to angle
        # This tells us if table is left, right or center
        # Xtion horizontal FOV is ~58 degrees
        # -------------------------------------------------------
        fov         = 58.0
        angle_deg   = (center_x - frame_width / 2) / (frame_width / 2) * (fov / 2)
        angle_rad   = np.deg2rad(angle_deg)

        # -------------------------------------------------------
        # Calculate goal position
        # Stop 0.5m before the table
        # -------------------------------------------------------
        goal_distance = max(0.0, depth - 0.5)
        goal_x        = goal_distance * np.cos(angle_rad)
        goal_y        = goal_distance * np.sin(angle_rad)

        rospy.loginfo(f"[INFO] Table depth: {depth:.2f}m | Goal distance: {goal_distance:.2f}m | Angle: {angle_deg:.1f}deg")

        # -------------------------------------------------------
        # Send goal to move_base
        # -------------------------------------------------------
        goal                           = MoveBaseGoal()
        goal.target_pose.header.frame_id   = 'base_link'  # relative to robot
        goal.target_pose.header.stamp      = rospy.Time.now()
        goal.target_pose.pose.position.x   = goal_x
        goal.target_pose.pose.position.y   = goal_y
        goal.target_pose.pose.position.z   = 0.0

        # Face towards the table
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = np.sin(angle_rad / 2)
        goal.target_pose.pose.orientation.w = np.cos(angle_rad / 2)

        self.navigating = True
        self.move_base_client.send_goal(
            goal,
            done_cb=self.goal_done_callback
        )
        rospy.loginfo("[INFO] Goal sent to move_base!")

    # -------------------------------------------------------
    # Called when move_base finishes navigation
    # -------------------------------------------------------
    def goal_done_callback(self, status, result):
        self.navigating = False
        rospy.loginfo("[INFO] Reached table! Stopping robot.")
        self.stop_robot()

    # -------------------------------------------------------
    # Control loop — runs every 100ms
    # -------------------------------------------------------
    def control_loop(self, event):
        # If table detected but not yet navigating, keep robot stopped
        if self.table_detected and not self.navigating:
            pass  # waiting for user input

    # -------------------------------------------------------
    # Stop the robot
    # -------------------------------------------------------
    def stop_robot(self):
        self.move_base_client.cancel_all_goals()
        twist         = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("[INFO] Robot STOPPED!")

    # -------------------------------------------------------
    # Main callback — runs every time RGB + Depth frames arrive
    # -------------------------------------------------------
    def image_callback(self, rgb_msg, depth_msg):
        rospy.loginfo("[DEBUG] Image received!")

        try:
            # Convert ROS messages to OpenCV
            frame       = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Convert depth to float32 in meters
            # Xtion gives depth in millimeters so divide by 1000
            depth_frame = depth_frame.astype(np.float32) / 1000.0

            # Run YOLO inference
            results = self.model.predict(source=frame, conf=0.25, verbose=False)

            masks = results[0].masks
            boxes = results[0].boxes
            names = results[0].names

            tables        = []  # list to store all detected tables
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            if masks is not None:
                for i, mask in enumerate(masks.data):
                    cls_id     = int(boxes[i].cls)
                    class_name = names[cls_id]
                    confidence = float(boxes[i].conf)

                    if class_name == "dining table":
                        # Convert mask to numpy and resize
                        mask_np      = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                        binary       = (mask_resized > 0.5).astype(np.uint8)

                        # -------------------------------------------------------
                        # Get depth of table
                        # Use median depth of masked region to avoid noise
                        # -------------------------------------------------------
                        depth_values = depth_frame[binary == 1]
                        depth_values = depth_values[depth_values > 0]  # remove zeros/invalid

                        if len(depth_values) == 0:
                            rospy.logwarn("[WARN] No valid depth for table, skipping!")
                            continue

                        table_depth = float(np.median(depth_values))

                        # Get center x of table (used for left-to-right ordering)
                        ys, xs = np.where(binary == 1)
                        center_x = int(np.mean(xs))
                        center_y = int(np.mean(ys))

                        tables.append({
                            'depth'  : table_depth,
                            'cx'     : center_x,
                            'cy'     : center_y,
                            'frame_w': frame.shape[1],
                            'mask'   : binary
                        })

                        # Add to combined mask for publishing
                        combined_mask = cv2.bitwise_or(combined_mask, binary * 255)

            # -------------------------------------------------------
            # Sort tables left to right by center_x
            # Table 1 = leftmost, Table 2 = next, etc.
            # -------------------------------------------------------
            tables = sorted(tables, key=lambda t: t['cx'])
            self.detected_tables = tables
            self.table_detected  = len(tables) > 0

            if len(tables) > 0:
                rospy.loginfo(f"[INFO] I can see {len(tables)} table(s) (left to right):")
                for idx, t in enumerate(tables):
                    rospy.loginfo(f"  Table {idx+1} | Depth: {t['depth']:.2f}m | CenterX: {t['cx']}px")
            else:
                rospy.loginfo("[DEBUG] No tables detected")

            # Publish binary mask
            mask_msg        = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            mask_msg.header = rgb_msg.header
            self.mask_pub.publish(mask_msg)

            # Publish annotated image
            annotated_frame      = results[0].plot(masks=True)
            annotated_msg        = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            annotated_msg.header = rgb_msg.header
            self.annotated_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"[ERROR] {e}")


if __name__ == '__main__':
    try:
        node = YoloSegNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

