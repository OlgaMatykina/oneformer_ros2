'''
export task=semantic

python demo/demo.py --config-file configs/valid/swin/oneformer_swin_large_sem_seg_bs4_640k.yaml \
  --input ../data/Integrant_validation_2023_10_13/* \
  --output outputs/valid_sem_seg_integrant \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS outputs/valid_sem_seg_v2/model_0009999.pth

'''
import rclpy
# import cv2

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# from semseg.semseg import SemanticSegmentator
from segformer_ros2.inference_speed_meter import InferenceSpeedMeter #перенести из semseg_ros2

import argparse
import multiprocessing as mp
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
# from detectron2.data.datasets import register_coco_instances

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo

from pycocotools.coco import COCO
from pathlib import Path

meta_clss = {0: 'unlabelled',
            1: 'firehose',
            2: 'hose',
            3: 'waste',
            4: 'puddle',
            5: 'breakroad',
            6: 'sidewalk',
            7: 'terrain',
            8: 'road'}

def valid_validation_register():
  list_of_annotations = []
  dataset_dir = "datasets/valid"
  annotation_dir = dataset_dir + "/" + "super_annotations_validation"
  images_dir = dataset_dir + "/" + "images_validation"
  coco = COCO('../data/val_super.json')
  for key in tqdm.tqdm(coco.imgs.keys()):

  #for file in tqdm.tqdm(list(annotation_dir.iterdir())):
      list_of_annotations.append(
          {
            "file_name": images_dir + "/" + coco.imgs[key]['file_name'],
            "height": int(coco.imgs[key]['height']),
            "width": int(coco.imgs[key]['width']),
            "image_id": coco.imgs[key]['id'],
            "sem_seg_file_name": annotation_dir + "/" + str(str(coco.imgs[key]['id']) + '.png'),
          }
      )
  return list_of_annotations


DatasetCatalog.register("valid_sem_seg_val", valid_validation_register)

MetadataCatalog.get("valid_sem_seg_val").stuff_classes = list(meta_clss.values())
MetadataCatalog.get("valid_sem_seg_val").ignore_label = 0
MetadataCatalog.get("valid_sem_seg_val").stuff_dataset_id_to_contiguous_id = {
    i: i
    for i in list(meta_clss.keys())
} 

MetadataCatalog.get("valid_sem_seg_val").evaluator_type = "sem_seg"


def valid_training_register():
  list_of_annotations = []
  dataset_dir = "datasets/valid"
  annotation_dir = dataset_dir + "/" + "super_annotations_training"
  images_dir = dataset_dir + "/" + "images_training"
  coco = COCO('../data/train_super.json')
  for key in tqdm.tqdm(coco.imgs.keys()):

  #for file in tqdm.tqdm(list(annotation_dir.iterdir())):
      list_of_annotations.append(
          {
            "file_name": images_dir + "/" + coco.imgs[key]['file_name'],
            "height": int(coco.imgs[key]['height']),
            "width": int(coco.imgs[key]['width']),
            "image_id": coco.imgs[key]['id'],
            "sem_seg_file_name": annotation_dir + "/" + str(str(coco.imgs[key]['id']) + '.png'),
          }
      )
  return list_of_annotations

DatasetCatalog.register("valid_sem_seg_train", valid_training_register)

MetadataCatalog.get("valid_sem_seg_train").stuff_classes = list(meta_clss.values())
MetadataCatalog.get("valid_sem_seg_train").ignore_label = 0
MetadataCatalog.get("valid_sem_seg_train").stuff_dataset_id_to_contiguous_id = {
    i: i
    for i in list(meta_clss.keys())
} 
MetadataCatalog.get("valid_sem_seg_train").thing_dataset_id_to_contiguous_id = {
    i: i
    for i in list(meta_clss.keys())
} 
MetadataCatalog.get("valid_sem_seg_train").thing_dataset_id_to_contiguous_id = {
    i: i
    for i in list(meta_clss.keys())
} 

MetadataCatalog.get("valid_sem_seg_train").evaluator_type = "sem_seg"

MetadataCatalog.get("valid_sem_seg_train").stuff_colors = [                    
                    (255,255,255), #'unlabeled' : none
                    (255,0,0), #'firehose' : red
                    (255,165,0), #'hose' : orange
                    (0,0,255), #'waste' : blue
                    (255,255,0), #'puddle' : yellow
                    (0,255,255), #'breakroad' : aqua
                    (255,0,255), #'sidewalk' : magenta
                    (0,128,0), #'terrain': green
                    (250,128,114) #'road' : salmon
                    ]

MetadataCatalog.get("valid_sem_seg_val").stuff_colors = [                    
                    (255,255,255), #'unlabeled' : none
                    (255,0,0), #'firehose' : red
                    (255,165,0), #'hose' : orange
                    (0,0,255), #'waste' : blue
                    (255,255,0), #'puddle' : yellow
                    (0,255,255), #'breakroad' : aqua
                    (255,0,255), #'sidewalk' : magenta
                    (0,128,0), #'terrain': green
                    (250,128,114) #'road' : salmon
                    ]


class SegFormerNode(Node):

    def setup_cfg(args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        return cfg

    def __init__(self) -> None:
        super().__init__('segformer_node')

        # self.declare_parameter('weights')
        # self.weights = self.get_parameter('weights').get_parameter_value().string_value

        self.declare_parameter('config-file')
        self.config_file = self.get_parameter('config-file').get_parameter_value().string_value

        self.declare_parameter('input')
        self.input = self.get_parameter('input').get_parameter_value().string_value

        self.declare_parameter('output')
        self.output = self.get_parameter('output').get_parameter_value().string_value

        self.declare_parameter('task')
        self.task = self.get_parameter('task').get_parameter_value().string_value

        self.declare_parameter('opts')
        self.opts = self.get_parameter('opts').get_parameter_value().string_value

        self.declare_parameter('confidence-threshold', 0.5)
        self.treshold = self.get_parameter('confidence-threshold').get_parameter_value().double_value

        #torch.cuda.set_device(1)
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        mp.set_start_method("spawn", force=True)
        # args = get_parser().parse_args()
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        # demo = VisualizationDemo(cfg)

        # self.segmentator = SemanticSegmentator(self.weights)
        self.segmentator = VisualizationDemo(cfg)

        self.br = CvBridge()

        self.sub_image = self.create_subscription(Image, 'image', self.on_image, 10)
        self.pub_segmentation = self.create_publisher(Image, 'segmentation', 10)

        self.speed_meter = InferenceSpeedMeter()

    


    def on_image(self, image_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        self.speed_meter.start()
        segmentation = self.segmentator.inference(image, self.treshold)
        self.speed_meter.stop()

        segmentation_msg = self.br.cv2_to_imgmsg(segmentation, 'mono8')
        segmentation_msg.header = image_msg.header

        self.pub_segmentation.publish(segmentation_msg)


        if args.input:
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation

                print(path)    
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img, args.task)
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )
                if args.output:
                    if len(args.input) == 1:
                        for k in visualized_output.keys():
                            print(k)
                            os.makedirs(k, exist_ok=True)
                            out_filename = os.path.join(k, args.output)
                            print(out_filename)
                            visualized_output[k].save(out_filename)    
                    else:
                        for k in visualized_output.keys():
                            opath = os.path.join(args.output, k)    
                            os.makedirs(opath, exist_ok=True)
                            out_filename = os.path.join(opath, os.path.basename(path))
                            visualized_output[k].save(out_filename)    
                else:
                    raise ValueError("Please specify an output path!")
        else:
            raise ValueError("No Input Given")

def main(args=None):
    rclpy.init(args=args)

    node = SegFormerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()