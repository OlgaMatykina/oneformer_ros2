import rclpy
import cv2
import numpy as np
import message_filters

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from semseg.oneform import SemanticSegmentator
from semseg.visualizer import ColorMode, Visualizer


class VisualizerNode(Node):

    def __init__(self):
        super().__init__('visualizer_node')

        image_sub = message_filters.Subscriber(self, Image, 'image')
        segmentation_sub = message_filters.Subscriber(self, Image, 'segmentation')

        self.ts = message_filters.TimeSynchronizer([image_sub, segmentation_sub], 10)
        self.ts.registerCallback(self.on_image_segmentation)

        self.pub_segmentation_color = self.create_publisher(Image, 'segmentation_color', 10)

        self.br = CvBridge()


    def on_image_segmentation(self, image_msg : Image, segm_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        segmentation = self.br.imgmsg_to_cv2(segm_msg, desired_encoding='mono8')

        visualizer = Visualizer(image, instance_mode=ColorMode.IMAGE_BW)

        segmentation_color = visualizer.draw_sem_seg(segmentation, alpha=0.7)
        segmentation_color.save('/home/docker_oneformer_ros2/colcon_ws/src/semseg_ros2/semseg_ros2/visualizer_images/3.jpg')
        segmentation_color = cv2.imread('/home/docker_oneformer_ros2/colcon_ws/src/semseg_ros2/semseg_ros2/visualizer_images/3.jpg')

        # segmentation_color = SemanticSegmentator.colorize(segmentation)
        # cv2.imwrite('/home/docker_oneformer_ros2/colcon_ws/src/semseg_ros2/semseg_ros2/visualizer_images/4.jpg', segmentation_color)

        segm_color_msg = self.br.cv2_to_imgmsg(segmentation_color, 'rgb8')
        segm_color_msg.header = segm_msg.header

        self.pub_segmentation_color.publish(segm_color_msg)


def main(args=None):
    rclpy.init(args=args)

    node = VisualizerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
