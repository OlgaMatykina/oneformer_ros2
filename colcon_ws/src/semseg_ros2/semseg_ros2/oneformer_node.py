import rclpy
import cv2

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from semseg.oneform import SemanticSegmentator
from semseg_ros2.inference_speed_meter import InferenceSpeedMeter


class SemSegNode(Node):

    def __init__(self) -> None:
        super().__init__('oneformer_node')

        self.declare_parameter('cfg')
        self.cfg = self.get_parameter('cfg').get_parameter_value().string_value

        self.declare_parameter('cat_num')
        self.cat_num = self.get_parameter('cat_num').get_parameter_value().integer_value

        # self.declare_parameter('treshold', 0.5)
        # self.treshold = self.get_parameter('treshold').get_parameter_value().double_value


        # print(self.cat_num)
        # print(type(self.cat_num))

        self.segmentator = SemanticSegmentator(self.cfg, self.cat_num)

        self.br = CvBridge()

        self.sub_image = self.create_subscription(Image, 'image', self.on_image, 10)
        self.pub_segmentation = self.create_publisher(Image, 'segmentation', 10)

        self.speed_meter = InferenceSpeedMeter()


    def on_image(self, image_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        self.speed_meter.start()
        segmentation = self.segmentator.inference(image)
        # print(segmentation)
        # cv2.imwrite('/home/docker_oneformer_ros2/colcon_ws/src/semseg_ros2/semseg_ros2/visualizer_images/2.jpg', segmentation)
        # cv2.imwrite('/home/docker_oneformer_ros2/colcon_ws/src/semseg_ros2/semseg_ros2/visualizer_images/2.jpg', self.segmentator.colorize(segmentation))
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
