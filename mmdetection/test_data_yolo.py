# Modified from: https://github.com/dimitymiller/openset_detection

import argparse
import os
import warnings
from pathlib import Path

import cv2
import torch
import torchvision.ops as ops

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops as yolo_ops

import numpy as np
import tqdm
import json
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    parser.add_argument('model_path', help='Path to object detector weights')
    parser.add_argument('imageset_path',  help='Path to ImageSets .txt file')
    parser.add_argument('--num_classes',  help='Number of total classes')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

args = parse_args()

# TODO: deprecated?
# Determine which os classes need to be removed from yolo test
suffix = args.saveNm[len("frcnn_GMMDet_Voc_"):] # custom -> xml, lru1 | yolo -> xml_yolo, lru1_yolo
suffix = suffix[:-len("_yolo")] # xml, lru1
num_classes_dict = { # Class IDs of ID classes
    'xml' : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    'lru1' : [0,1,2],
    'lru1_drone' : [1,2],
    'lru1_lander' : [0,2],
    'lru1_lru2' : [0,1],
}
id_classes = num_classes_dict[suffix] # CS classes

###################################################################################################
############## Setup Config #######################################################################
###################################################################################################
class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

def load_and_prepare_model(model_path):
    """
    Load YOLO model and register forward hooks.
    Hook intermediate features and raw predictions to reverse-engineer YOLO model outputs.

    Args:
        model_path (YOLO): Path to YOLOv8 model

    Returns:
        model: YOLO model.
        hooks: List of registered hook references
    """
    # Load YOLO model
    model = YOLO(model_path)
    detect = None
    cv2_hooks = None
    cv3_hooks = None
    detect_hook = SaveIO()

    # Identify Detect layer (YOLO head) and register forward hook (`detect_hook`)
    for i, module in enumerate(model.model.modules()):
        if type(module) is Detect:
            module.register_forward_hook(detect_hook)
            detect = module

            # Register forward hooks on detection scale's internal convolution layers (`cv2` and `cv3`)
            cv2_hooks = [SaveIO() for _ in range(module.nl)]
            cv3_hooks = [SaveIO() for _ in range(module.nl)]
            for i in range(module.nl):
                module.cv2[i].register_forward_hook(cv2_hooks[i])
                module.cv3[i].register_forward_hook(cv3_hooks[i])
            break
    input_hook = SaveIO()

    # Register top-level forward hook on entire model
    model.model.register_forward_hook(input_hook)
    hooks = [input_hook, detect, detect_hook, cv2_hooks, cv3_hooks]

    return model, hooks

def run_predict(img, img_path, model, hooks, num_classes, conf_threshold=0.2, iou_threshold=0.5):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        num_classes (int): Number of os classes.
        conf_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
    # Unpack hooks from load_and_prepare_model()
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    # Run inference. Results are stored by hooks
    model(img, verbose=False)

    # Reverse engineer outputs to find logits
    # See Detect.forward(): https://github.com/ultralytics/ultralytics/blob/b638c4ed9a24270a6875cdd47d9eeda99204ef5a/ultralytics/nn/modules/head.py#L22
    shape = detect_hook.input[0][0].shape  # BCHW
    x = []
    for i in range(detect.nl):
        x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
    x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
    box, classes = x_cat.split((detect.reg_max * 4, detect.nc), 1)

    # Assume batch size 1 (i.e. just running with one image)
    # Loop here to batch images
    batch_idx = 0
    xywh_sigmoid = detect_hook.output[0][batch_idx]
    all_logits = classes[batch_idx]

    # Get original image shape and model image shape to transform boxes
    img_shape = input_hook.input[0].shape[2:]
    orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

    # Compute predictions
    boxes = []
    for i in range(xywh_sigmoid.shape[-1]): # for each predicted box...
        x0, y0, x1, y1, *class_probs_after_sigmoid = xywh_sigmoid[:,i]
        x0, y0, x1, y1 = yolo_ops.scale_boxes(img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), orig_img_shape)
        logits = all_logits[:,i]
        
        boxes.append({
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
            'logits': logits.cpu().tolist(),
            'activations': [p.item() for p in class_probs_after_sigmoid]
        })

    # Non-Maximum-Suppresion (Retain only the most relevant boxes based on confidence scores)
    boxes_for_nms = torch.stack([
        torch.tensor([
            *b['bbox_xywh'],
            *b['activations'],
            *b['activations'],
            *b['logits']]) for b in boxes
    ], dim=1).unsqueeze(0)

    nms_results_batch = yolo_ops.non_max_suppression(
        boxes_for_nms,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        nc=detect.nc
    )

    # Initialize final output: one list per class
    class_results = [[] for _ in range(num_classes)]

    for nms_results in nms_results_batch:
        if nms_results is None or nms_results.shape[0] == 0:
            continue

        for b in range(nms_results.shape[0]):
            box = nms_results[b, :]
            x0, y0, x1, y1, conf, cls, *acts_and_logits = box
            logits = acts_and_logits[detect.nc:]
            cls_idx = int(cls.item())
            bbox = [x0.item(), y0.item(), x1.item(), y1.item()]  # xyxy
            score = conf.item()
            logits = [p.item() for p in logits]

            # Pack: [x1, y1, x2, y2, score, logit_0, ..., logit_C]
            row = bbox + [score] + logits
            class_results[cls_idx].append(row)

    # Convert each class list to np.ndarray of shape (k_i, 5+num_classes)
    class_results = [
        np.array(cls_boxes, dtype=np.float32) if cls_boxes else np.zeros((0, 5 + num_classes), dtype=np.float32)
        for cls_boxes in class_results
    ]

    return class_results

###################################################################################################
############### Load Dataset ######################################################################
###################################################################################################
print("Building datasets")

# Preload images (keep original path and loaded image)
imageset_file = Path(args.imageset_path)
image_paths = [Path(line.strip()) for line in imageset_file.read_text().splitlines()] # Read lines, strip whitespace, and convert to Path objects
preloaded_images = [(p, cv2.imread(str(p))) for p in image_paths]

###################################################################################################
############### Build model #######################################################################
###################################################################################################
print("Building model")

# Load model
model, hooks = load_and_prepare_model(args.model_path)

########################################################################################################
########################## TESTING DATA  ###############################################################
########################################################################################################
print(f"Testing {args.imageset_path} data")

score_threshold = 0.2 # only detections with a max softmax above this score are considered valid
iou_threshold = 0.5
total = 0
allResults = {}
for image_path, img in tqdm.tqdm(preloaded_images, total=len(preloaded_images)):
    total += 1

    imName = image_path.name
    imName = "JPEGImages/" + imName 
    allResults[imName] = []
    all_detections = None
    all_scores = []
    
    # Predict logits using image and path
    result = run_predict(img, image_path, model, hooks, int(args.num_classes), conf_threshold=score_threshold, iou_threshold=iou_threshold)

    #collect results from each class and concatenate into a list of all the results
    for j in range(np.shape(result)[0]):
        dets = result[j]

        if len(dets) == 0:
            continue

        bboxes = dets[:, :4] # column 0-3
        dists = dets[:, 5:]  # Logits from column index 5
        scores = dets[:, 4] # column 4
        scoresT = np.expand_dims(scores, axis=1)

        #if len(dists) > 0:
        #    print(imName)
        #    print(f'Winning class {j}')
        #    print(dists)

        # !!! WE ARE NOT ADDING THE BACKRGOUND LOGIT
        # Add background logit. Note Yolov8 uses a sigmoid head (instead of softmax). Classes are assumed to be independent.
        # Convert to PyTorch tensor
        #dists_tensor = torch.from_numpy(dists).float()  # [N, C]
        # Compute sigmoid probabilities per class
        #sigmoid_probs = torch.sigmoid(dists_tensor)  # [N, C]
        # Conservative estimate of background probability: 1 - max class prob
        #p_bg = 1 - torch.max(sigmoid_probs, dim=1).values.clamp(min=1e-6, max=1 - 1e-6)  # [N]
        # Compute sum of exp(logits) for foreground classes (softmax denominator)
        #sum_exp = torch.exp(dists_tensor).sum(dim=1)  # [N]
        # Compute pseudo background logit using softmax math
        #bg_logits = torch.log((p_bg / (1 - p_bg)) * sum_exp).unsqueeze(1)  # [N, 1]
        # Concatenate background logit after foreground logits
        #frcnn_logits = torch.cat([dists_tensor, bg_logits], dim=1).numpy()  # [N, C+1], NumPy array

        #winning class must be class j for this detection to be considered valid
        mask = np.argmax(dists, axis = 1)==j

        # This check needs to be done after mask with j,because YOLO trains on all classes and uses indices according to that.
        # dists = dists[:, id_classes] # Only keep id classes

        if np.sum(mask) == 0:
            continue

        #check thresholds are above the score cutoff
        imDets = np.concatenate((dists, bboxes, scoresT), 1)[mask]
        scores = scores[mask]
        mask2 = scores >= score_threshold

        if np.sum(mask2) == 0:
            continue
        
        imDets = imDets[mask2]

        if all_detections is None:
            all_detections = imDets
        else:
            all_detections = np.concatenate((all_detections, imDets))

    if all_detections is None:
        continue
    else:
        #remove doubled-up detections -- this shouldn't really happen
        detections, idxes = np.unique(all_detections, return_index = True, axis = 0)

    allResults[imName] = detections.tolist()

#save results
jsonRes = json.dumps(allResults)

save_dir = f'{BASE_RESULTS_FOLDER}/YOLOv8/raw/custom/{imageset_file.stem}'
#check folders exist, if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open('{}/{}.json'.format(save_dir, args.saveNm), 'w')
f.write(jsonRes)
f.close()

