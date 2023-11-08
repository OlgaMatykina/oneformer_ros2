import torch
import argparse
import numpy as np

from torchvision.models.segmentation import fcn_resnet50


class SemanticSegmentator:

    def __init__(self, weights):
        # self.model = fcn_resnet50(weights='DEFAULT')
        self.model = fcn_resnet50(pretrained=False, pretrained_backbone=False, aux_loss=True)
        state_dict = torch.load(weights)
        self.model.load_state_dict(state_dict)

        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def inference(self, image, treshold=0.5):
        batch = SemanticSegmentator.to_tensor(image)

        logits = self.model(batch)

        probs = torch.softmax(logits['aux'][0], 0)
        segmentation = probs.argmax(dim=0) * (probs.max(dim=0).values > treshold)

        return SemanticSegmentator.to_ndarray(segmentation)


    @staticmethod
    def to_tensor(image : np.ndarray):
        image_tensor = torch.Tensor(image.copy()).float() / 255
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            mean = mean.cuda()
            std = std.cuda()

        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.permute(2, 0, 1)

        batch = image_tensor.unsqueeze(0)

        return batch


    @staticmethod
    def to_ndarray(segmentation : torch.Tensor):
        return segmentation.cpu().numpy().astype(np.uint8)
    

    @staticmethod
    def colorize(segmentation : np.ndarray):
        pallete = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 255],  # bicycle
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [255, 0, 0],  # car
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 255, 0],  # person
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        segmentation_color = pallete[segmentation]

        return segmentation_color
