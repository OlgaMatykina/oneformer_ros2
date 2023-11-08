import rclpy
import cv2

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from semseg.semseg import SemanticSegmentator #from predictor import VisualizationDemo
from semseg_ros2.inference_speed_meter import InferenceSpeedMeter


class SemSegNode(Node):

    def __init__(self) -> None:
        super().__init__('semseg_node')

        self.declare_parameter('weights')
        self.weights = self.get_parameter('weights').get_parameter_value().string_value

        self.declare_parameter('treshold', 0.5)
        self.treshold = self.get_parameter('treshold').get_parameter_value().double_value

        self.segmentator = SemanticSegmentator(self.weights) #demo = VisualizationDemo(cfg)

        self.br = CvBridge()

        self.sub_image = self.create_subscription(Image, 'image', self.on_image, 10)
        self.pub_segmentation = self.create_publisher(Image, 'segmentation', 10)

        self.speed_meter = InferenceSpeedMeter()


    def on_image(self, image_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        self.speed_meter.start()
        segmentation = self.segmentator.inference(image, self.treshold) #predictions, visualized_output = demo.run_on_image(img, args.task)
        self.speed_meter.stop()

        segmentation_msg = self.br.cv2_to_imgmsg(segmentation, 'mono8')
        segmentation_msg.header = image_msg.header

        self.pub_segmentation.publish(segmentation_msg)


def main(args=None):
    rclpy.init(args=args)

    node = SemSegNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()