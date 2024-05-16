import os
import random
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (scale * channel - mean) / std

    Args:
        scale (float): Scaling constant.
        mean (sequence): Sequence of means for R,G,B channels respecitvely.
        std (sequence): Sequence of standard deviations for R,G,B channels
            respecitvely.
        depth_scale (float): Depth divisor for depth annotations.

    """

    def __init__(self, scale, mean, std, depth_scale=1.0):
        self.scale = scale
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)
        self.depth_scale = depth_scale

    def __call__(self, sample):
        sample["image"] = (self.scale * sample["image"].to(device) - self.mean) / self.std
        if "depth" in sample:
            sample["depth"] = sample["depth"].to(device) / self.depth_scale
        return sample
        
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["masks"]
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = torch.random.randint(0, h - new_h + 1)
        left = torch.random.randint(0, w - new_w + 1)
        sample["image"] = image[top : top + new_h, left : left + new_w]
        for msk_key in msk_keys:
            sample[msk_key] = sample[msk_key][top : top + new_h, left : left + new_w]
        return sample
        
# Usual dtypes for common modalities
KEYS_TO_DTYPES = {
    "segm": torch.long,
    "mask": torch.long,
    "depth": torch.float,
    "normals": torch.float,
}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["masks"]
        sample["image"] = torch.from_numpy(image.transpose((2, 0, 1))).to(device)
        for msk_key in msk_keys:
            sample[msk_key] = torch.from_numpy(sample[msk_key]).to(KEYS_TO_DTYPES[msk_key]).to(device)
        return sample
        
class RandomMirror(object):
    """Randomly flip the image and the mask"""

    def __call__(self, sample):
        image = sample["image"].to(device)
        msk_keys = sample["masks"]
        do_mirror = torch.random.randint(2, device=device)
        if do_mirror:
            sample["image"] = cv2.flip(image.cpu().numpy(), 1).to(device)
            for msk_key in msk_keys:
                scale_mult = torch.tensor([-1, 1, 1], device=device) if "normal" in msk_key else 1
                sample[msk_key] = (scale_mult * cv2.flip(sample[msk_key].cpu().numpy(), 1)).to(device)
        return sample


class AverageMeter:
    """Simple running average estimator.
    Args:
      momentum (float): running average decay.
    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.avg = torch.tensor(0.0, device=device)
        self.val = None


    def update(self, val):
        """Update running average given a new value.
        The new running average estimate is given as a weighted combination \
        of the previous estimate and the current value.
        Args:
          val (float): new value
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = (self.avg * self.momentum + val * (1.0 - self.momentum))
        self.val = val
    
def fast_cm(preds, gt, n_classes):
    """Computing confusion matrix faster.
    Args:
      preds (Tensor) : predictions (either flatten or of size (len(gt), top-N)).
      gt (Tensor) : flatten gt.
      n_classes (int) : number of classes.
    Returns:
      Confusion matrix (Tensor of size (n_classes, n_classes)).
    """

    cm = torch.zeros((n_classes, n_classes), dtype=torch.int, device=device)

    #print(gt.shape)
    #i,a,p, n = gt.shape[0]

    for i in range(gt.shape[0]):
        a = gt[i].long().to(device)
        p = preds[i].long().to(device)
        cm[a, p] += 1
    return cm.to(device)
    
def compute_iu(cm):
    """Compute IU from confusion matrix.
    Args:
      cm (Tensor) : square confusion matrix.
    Returns:
      IU vector (Tensor).
    """
    pi = 0
    gi = 0
    ii = 0
    denom = 0
    n_classes = cm.shape[0]
    # IU is between 0 and 1, hence any value larger than that can be safely ignored
    default_value = 2
    IU = (torch.ones(n_classes) * default_value).to(device)
    for i in range(n_classes):
        pi = sum(cm[:, i])
        gi = sum(cm[i, :])
        ii = cm[i, i]
        denom = pi + gi - ii
        if denom > 0:
            IU[i] = ii / denom
    return IU.to(device)


class MeanIoU:
    """Mean-IoU computational block for semantic segmentation.
    Args:
      num_classes (int): number of classes to evaluate.
    Attributes:
      name (str): descriptor of the estimator.
    """

    def __init__(self, num_classes):
        
        if isinstance(num_classes, (list, tuple)):
            num_classes = num_classes[0]
        assert isinstance( num_classes, int), f"Number of classes must be int, got {num_classes}"
        self.num_classes = num_classes
        self.name = "meaniou"
        self.reset()

    def reset(self):
        
        self.cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int, device=device)

    def update(self, pred, gt):
        
        idx = gt < self.num_classes
        pred_dims = len(pred.shape)
        assert (pred_dims - 1) == len( gt.shape), "Prediction tensor must have 1 more dimension that ground truth"
        
        if pred_dims == 3:
            class_axis = 0
        elif pred_dims == 4:
            class_axis = 1
        else:
            raise ValueError("{}-dimensional input is not supported".format(pred_dims))
        assert (
            pred.shape[class_axis] == self.num_classes
        ), "Dimension {} of prediction tensor must be equal to the number of classes".format(
            class_axis
        )
        pred = pred.argmax(axis=class_axis).to(device)
        gt = gt.to(device)
        # print("Prediction shape",pred.shape)
        # print("Prediction dime",pred_dims)
        # print("ground truth",gt.shape)
        # print("number of classes",self.num_classes)
        self.cm += fast_cm(pred[idx], gt[idx], self.num_classes)

    def val(self):
        
        ious = compute_iu(self.cm)
        return torch.mean([iu for iu in ious if iu <= 1.0]).to(device)
        

class RMSE:
    """Root Mean Squared Error computational block for depth estimation.
    Args:
      ignore_val (float): value to ignore in the target
                          when computing the metric.
    Attributes:
      name (str): descriptor of the estimator.
    """

    def __init__(self, ignore_val=0):
        self.ignore_val = ignore_val
        self.name = "rmse"
        self.reset()

    def reset(self):
        self.num = torch.tensor(0.0, device=device)
        self.den = torch.tensor(0.0, device=device)

    def update(self, pred, gt):
        assert (pred.shape == gt.shape), f"Prediction tensor {pred.shape} must have the same shape as ground truth gt{gt.shape}"
        pred = torch.abs(pred[1:]).to(device)
        gt = gt.to(device)
        idx = gt != self.ignore_val
        diff = (pred - gt)[idx]
        self.num += torch.sum(diff ** 2)
        self.den += torch.sum(idx)

    def val(self):
        return torch.sqrt(self.num / self.den)


class InvHuberLoss(nn.Module):
    """Inverse Huber Loss for depth estimation.
    The setup is taken from https://arxiv.org/abs/1606.00373
    Args:
      ignore_index (float): value to ignore in the target
                            when computing the loss.
    """

    def __init__(self, ignore_index=0):
        super(InvHuberLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, target):
        x = x.to(device)
        target = target.to(device)
        input = F.relu(x)  # depth predictions must be >=0
        diff = input - target
        mask = target != self.ignore_index

        err = torch.abs(diff * mask.float())
        c = 0.2 * torch.max(err)
        err2 = (diff ** 2 + c ** 2) / (2.0 * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = torch.mean(err * mask_err.float() + err2 * mask_err2.float())
        return cost

