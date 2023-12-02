import torch

import numpy as np

import multiprocessing as mp

from detectron2.data import MetadataCatalog
from semseg.defaults import DefaultPredictor
from detectron2.config import get_cfg
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from detectron2.projects.deeplab import add_deeplab_config
# from visualizer import ColorMode, Visualizer

# from torchvision.models.segmentation import fcn_resnet50


class SemanticSegmentator:

    def __init__(self, config_file, cat_num):
        # meta_clss = {0: 'unlabeled',
        #     1: 'firehose',
        #     2: 'hose',
        #     3: 'waste',
        #     4: 'puddle',
        #     5: 'breakroad',
        #     6: 'sidewalk',
        #     7: 'terrain',
        #     8: 'road'}
        print(cat_num)
        print(type(cat_num))
        self.cat_num=cat_num
        if self.cat_num==10:
            meta_clss = {0: 'unlabelled',
                1: 'firehose',
                2: 'hose',
                3: 'waste',
                4: 'puddle',
                5: 'breakroad',
                6: 'sidewalk',
                7: 'terrain',
                8: 'vegetation',
                9: 'road'}
        elif self.cat_num==6:
            meta_clss = {0: 'unlabelled',
                1: 'breakroad',
                2: 'sidewalk',
                3: 'terrain',
                4: 'vegetation',
                5: 'road'}
        else:
            print("Enter number of categories")
            
        MetadataCatalog.get("valid_sem_seg_val").stuff_classes = list(meta_clss.values())
        MetadataCatalog.get("valid_sem_seg_val").ignore_label = 255
        MetadataCatalog.get("valid_sem_seg_val").stuff_dataset_id_to_contiguous_id = {
            i: i
            for i in list(meta_clss.keys())
        } 
        MetadataCatalog.get("valid_sem_seg_val").thing_dataset_id_to_contiguous_id = {
            i: i
            for i in list(meta_clss.keys())
        }
        # MetadataCatalog.get("valid_sem_seg_val").stuff_colors = [                    
        #             (255,255,255), #'unlabeled' : none
        #             (255,0,0), #'firehose' : red
        #             (255,165,0), #'hose' : orange
        #             (0,0,255), #'waste' : blue
        #             (255,255,0), #'puddle' : yellow
        #             (0,255,255), #'breakroad' : aqua
        #             (255,0,255), #'sidewalk' : magenta
        #             (0,128,0), #'terrain': green
        #             (250,128,114) #'road' : salmon
        #             ]
        if self.cat_num==10:
            MetadataCatalog.get("valid_sem_seg_val").stuff_colors = [                    
                        (255,255,255), #'unlabelled' : none
                        (255,0,0), #'firehose' : red
                        (255,165,0), #'hose' : orange
                        (0,0,255), #'waste' : blue
                        (255,255,0), #'puddle' : yellow
                        (0,255,255), #'breakroad' : aqua
                        (255,0,255), #'sidewalk' : magenta
                        (0,128,0), #'terrain': green
                        (127,72,41), #'vegetation': brown
                        (250,128,114) #'road' : salmon
                        ]
        elif self.cat_num==6:
            MetadataCatalog.get("valid_sem_seg_val").stuff_colors = [                    
                        (255,255,255), #'unlabelled' : none
                        (0,255,255), #'breakroad' : aqua
                        (255,0,255), #'sidewalk' : magenta
                        (0,128,0), #'terrain': green
                        (127,72,41), #'vegetation': brown
                        (250,128,114) #'road' : salmon
                        ]
        else:
            print("Enter number of categories")

        self.metadata = MetadataCatalog.get("valid_sem_seg_val")
        
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        cfg.merge_from_file(config_file)
        self.cpu_device = torch.device("cpu")
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)


    def inference(self, image):
        
        # batch = SemanticSegmentator.to_tensor(image)
        image = image[:, :, ::-1]

        probs = self.predictor(image, 'semantic')["sem_seg"]

        # probs = torch.softmax(logits['aux'][0], 0)
        segmentation = probs.argmax(dim=0) # * (probs.max(dim=0).values > treshold)
        # print(segmentation)


        return SemanticSegmentator.to_ndarray(segmentation)


    # @staticmethod
    # def to_tensor(image : np.ndarray):
    #     image_tensor = torch.Tensor(image.copy()).float() / 255
    #     mean = torch.Tensor([0.485, 0.456, 0.406])
    #     std = torch.Tensor([0.229, 0.224, 0.225])

    #     if torch.cuda.is_available():
    #         image_tensor = image_tensor.cuda()
    #         mean = mean.cuda()
    #         std = std.cuda()

    #     image_tensor = (image_tensor - mean) / std
    #     image_tensor = image_tensor.permute(2, 0, 1)

    #     batch = image_tensor.unsqueeze(0)

    #     return batch


    @staticmethod
    def to_ndarray(segmentation : torch.Tensor):
        return segmentation.cpu().numpy().astype(np.uint8)
    

    @staticmethod
    def colorize(segmentation : np.ndarray):
        # pallete = np.array([
        #     [255,255,255], #'unlabeled' : none
        #     [255,0,0], #'firehose' : red
        #     [255,165,0], #'hose' : orange
        #     [0,0,255], #'waste' : blue
        #     [255,255,0], #'puddle' : yellow
        #     [0,255,255], #'breakroad' : aqua
        #     [255,0,255], #'sidewalk' : magenta
        #     [0,128,0], #'terrain': green
        #     [250,128,114] #'road' : salmon
        # ], dtype=np.uint8)
        pallete = np.array([
            [255,255,255], #'unlabeled' : none
            [255,0,0], #'firehose' : red
            [255,165,0], #'hose' : orange
            [0,0,255], #'waste' : blue
            [255,255,0], #'puddle' : yellow
            [0,255,255], #'breakroad' : aqua
            [255,0,255], #'sidewalk' : magenta
            [0,128,0], #'terrain': green
            [127,72,41], #'vegetation': brown
            [250,128,114] #'road' : salmon
        ], dtype=np.uint8)

        segmentation_color = pallete[segmentation]

        return segmentation_color



# class VisualizationDemo(object):
#     def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
#         """
#         Args:
#             cfg (CfgNode):
#             instance_mode (ColorMode):
#             parallel (bool): whether to run the model in different processes from visualization.
#                 Useful since the visualization logic can be slow.
#         """
#         self.metadata = MetadataCatalog.get(
#             cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
#         )
#         if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
#             from cityscapesscripts.helpers.labels import labels
#             stuff_colors = [k.color for k in labels if k.trainId != 255]
#             self.metadata = self.metadata.set(stuff_colors=stuff_colors)
#         self.cpu_device = torch.device("cpu")
#         self.instance_mode = instance_mode

#         self.parallel = parallel
#         if parallel:
#             num_gpu = torch.cuda.device_count()
#             self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
#         else:
#             self.predictor = DefaultPredictor(cfg)

#     def run_on_image(self, image, task):
#         """
#         Args:
#             image (np.ndarray): an image of shape (H, W, C) (in BGR order).
#                 This is the format used by OpenCV.
#         Returns:
#             predictions (dict): the output of the model.
#             vis_output (VisImage): the visualized image output.
#         """
#         vis_output = None
#         # Convert image from OpenCV BGR format to Matplotlib RGB format.
#         image = image[:, :, ::-1]
#         vis_output = {}
        
#         if task == 'panoptic':
#             visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE)
#             predictions = self.predictor(image, task)
#             panoptic_seg, segments_info = predictions["panoptic_seg"]
#             vis_output['panoptic_inference'] = visualizer.draw_panoptic_seg_predictions(
#             panoptic_seg.to(self.cpu_device), segments_info, alpha=0.7
#         )

#         if task == 'panoptic' or task == 'semantic':
#             visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE_BW)
#             predictions = self.predictor(image, task)
#             vis_output['semantic_inference'] = visualizer.draw_sem_seg(
#                 predictions["sem_seg"].argmax(dim=0).to(self.cpu_device), alpha=0.7
#             )

#         if task == 'panoptic' or task == 'instance':
#             visualizer = Visualizer(image, metadata=self.metadata, instance_mode=ColorMode.IMAGE_BW)
#             predictions = self.predictor(image, task)
#             instances = predictions["instances"].to(self.cpu_device)
#             vis_output['instance_inference'] = visualizer.draw_instance_predictions(predictions=instances, alpha=1)

#         return predictions, vis_output
