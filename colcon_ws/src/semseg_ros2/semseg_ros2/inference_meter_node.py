import pickle
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# from yolov8_seg_interfaces.msg import Objects


class InferenceMeterNode(Node):
    def __init__(self) -> None:
        super().__init__("oneformer_inf_node")

        self.declare_parameter("queue_size", 10)
        self.queue_size = (
            self.get_parameter("queue_size").get_parameter_value().integer_value
        )

        self.get_logger().info("Init segmentator")
        
        self.sub_camera = self.create_subscription(
            Image, "/sensum/left/image_raw", self.on_image, self.queue_size
        )
        self.sub_segmentator = self.create_subscription(
            Image, "/sensum/left/segmentation", self.on_segm, self.queue_size
        )
        self.sub_visualizer = self.create_subscription(
            Image, "/sensum/left/segmentation_color", self.on_vis, self.queue_size
        )

        self.messages = {}

    #  std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1688115518, nanosec=101541266), frame_id='cam_left')

    def on_image(self, image_msg: Image):
        header_time = image_msg.header.stamp
        key = f"s_{header_time.sec}_ns_{header_time.nanosec}"
        if self.messages.get(key, None) is None:
            self.messages[key] = {"cam_time": time.time()}
        else:
            self.messages[key]["cam_time"] = time.time()

        self.save_stats()

    def on_segm(self, segm_msg: Image):
        header_time = segm_msg.header.stamp
        key = f"s_{header_time.sec}_ns_{header_time.nanosec}"
        if self.messages.get(key, None) is None:
            self.messages[key] = {"seg_time": time.time()}
        else:
            self.messages[key]["seg_time"] = time.time()

        self.save_stats()

    def on_vis(self, vis_msg: Image):
        header_time = vis_msg.header.stamp
        key = f"s_{header_time.sec}_ns_{header_time.nanosec}"
        if self.messages.get(key, None) is None:
            self.messages[key] = {"vis_time": time.time()}
        else:
            self.messages[key]["vis_time"] = time.time()

        self.save_stats()

    def save_stats(self):
        with open("cache.pickle", "wb") as file:
            pickle.dump(self.messages, file)

def main(args=None):
    rclpy.init(args=args)

    node = InferenceMeterNode()
    node.get_logger().info("InferenceMeter node is ready")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()