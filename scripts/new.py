#!/usr/bin/env python3

# ROS libraries
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import actionlib
from move_base_msgs.msg import MoveBaseAction

# Vision libraries
import cv2
import numpy as np
from ultralytics import YOLO


class YoloSegNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolo_seg_node', anonymous=True)

        # -------------------------------------------------------
        # Load YOLO segmentation model
        # -------------------------------------------------------
        rospy.loginfo("[INFO] Loading YOLO model...")
        self.model = YOLO('yolov8x-seg.pt')
        rospy.loginfo("[INFO] Model loaded!")

        # CvBridge converts between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

        # Flag to track if table is detected
        self.table_detected = False

        # -------------------------------------------------------
        # Connect to move_base action server
        # This allows us to cancel navigation goals
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
        # Subscriber — listens to RGB camera topic
        # -------------------------------------------------------
        self.image_sub = rospy.Subscriber(
            '/xtion/rgb/image_raw',
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        # -------------------------------------------------------
        # Timer — keeps publishing stop command every 100ms
        # while table is detected so move_base cant override it
        # -------------------------------------------------------
        rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("[INFO] Node started! Waiting for images on /xtion/rgb/image_raw ...")

    # -------------------------------------------------------
    # Control loop — runs every 100ms
    # Keeps stopping robot as long as table is visible
    # -------------------------------------------------------
    def control_loop(self, event):
        if self.table_detected:
            self.stop_robot()

    # -------------------------------------------------------
    # Stop the robot
    # 1. Cancel move_base navigation goal
    # 2. Publish zero velocity
    # -------------------------------------------------------
    def stop_robot(self):
        # Cancel navigation goal so move_base stops sending commands
        self.move_base_client.cancel_all_goals()

        # Publish zero velocity to fully stop the robot
        twist = Twist()
        twist.linear.x  = 0.0
        twist.linear.y  = 0.0
        twist.linear.z  = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("[INFO] Table detected — Navigation CANCELLED and Robot STOPPED!")

    # -------------------------------------------------------
    # Main callback — runs every time a new image is received
    # -------------------------------------------------------
    def image_callback(self, msg):
        rospy.loginfo("[DEBUG] Image received!")

        try:
            # Convert ROS Image message to OpenCV BGR image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # -------------------------------------------------------
            # Run YOLO inference
            # -------------------------------------------------------
            results = self.model.predict(source=frame, conf=0.25, verbose=False)
            rospy.loginfo(f"[DEBUG] Inference done | Detections: {len(results[0].boxes)}")

            # Get masks, boxes, and class names from results
            masks = results[0].masks
            boxes = results[0].boxes
            names = results[0].names

            table_found = False

            # Empty black mask — will fill white where table is
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # -------------------------------------------------------
            # Loop through all detections and check for table
            # -------------------------------------------------------
            if masks is not None:
                for i, mask in enumerate(masks.data):
                    cls_id     = int(boxes[i].cls)
                    class_name = names[cls_id]
                    confidence = float(boxes[i].conf)

                    rospy.loginfo(f"[DEBUG] Detected: {class_name} | Confidence: {confidence:.2f}")

                    # COCO dataset uses 'dining table' for tables
                    if class_name == "dining table":
                        table_found = True
                        rospy.loginfo("[INFO] TABLE FOUND!")

                        # Convert mask tensor to numpy
                        mask_np = mask.cpu().numpy()

                        # Resize mask to original frame size
                        mask_resized = cv2.resize(
                            mask_np,
                            (frame.shape[1], frame.shape[0])
                        )

                        # Binary mask — 255 = table, 0 = background
                        binary = (mask_resized > 0.5).astype(np.uint8) * 255

                        # Combine masks if multiple tables found
                        combined_mask = cv2.bitwise_or(combined_mask, binary)

            # -------------------------------------------------------
            # Update table detection flag
            # control_loop will handle stopping the robot
            # -------------------------------------------------------
            self.table_detected = table_found

            if not table_found:
                rospy.loginfo("[DEBUG] No table detected — Robot free to move")

            # -------------------------------------------------------
            # Publish binary mask (viewable in RViz)
            # white = table, black = background
            # -------------------------------------------------------
            mask_msg        = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            # -------------------------------------------------------
            # Publish annotated image with YOLO overlay (viewable in RViz)
            # -------------------------------------------------------
            annotated_frame      = results[0].plot(masks=True)
            annotated_msg        = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"[ERROR] {e}")


if __name__ == '__main__':
    try:
        node = YoloSegNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass