import json
import os
from pathlib import Path
import argparse

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as ops
from PIL import Image
import tqdm
import random

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops as yolo_ops

# https://gist.github.com/justinkay/8b00a451b1c1cc3bcf210c86ac511f46
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

def run_predict_single(img_path, model, hooks, num_classes, conf_threshold=0.5, iou_threshold=0.7):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        num_classes (int): Number of classes.
        conf_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for NMS. Default is 0.7.

    Returns:
        list: List of selected bounding box dictionaries after NMS.
    """
    # Unpack hooks from load_and_prepare_model()
    input_hook, detect, detect_hook, cv2_hooks, cv3_hooks = hooks

    # Run inference. Results are stored by hooks
    model(img_path, verbose=False)

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
            'image_id': img_path,
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()], # xyxy
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
            'logits': logits.cpu().tolist(),
            'activations': [p.item() for p in class_probs_after_sigmoid]
        })

    # Debugging
    # top10 = sorted(boxes, key=lambda x: max(x['activations']), reverse=True)[:10]
    # plot_image(img_path, top10, suffix="before_nms")

    # NMS
    # We can keep the activations and logits around via the YOLOv8 NMS method, but only if we
    # append them as an additional time to the prediction vector. It's a weird hacky way to do it,
    # but it works. We also have to pass in the num classes (nc) parameter to make it work.
    boxes_for_nms = torch.stack([
        torch.tensor([
            *b['bbox_xywh'],
            *b['activations'],
            *b['activations'],
            *b['logits']]) for b in boxes
    ], dim=1).unsqueeze(0)
    nms_results = yolo_ops.non_max_suppression(boxes_for_nms, conf_thres=conf_threshold, iou_thres=iou_threshold, nc=detect.nc)[0]
    
    # Unpack and return
    boxes = []
    for b in range(nms_results.shape[0]):
        box = nms_results[b, :]
        x0, y0, x1, y1, conf, cls, *acts_and_logits = box
        activations = acts_and_logits[:detect.nc]
        logits = acts_and_logits[detect.nc:]
        box_dict = {
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()], # xyxy
            'bbox_xywh': [(x0.item() + x1.item())/2, (y0.item() + y1.item())/2, x1.item() - x0.item(), y1.item() - y0.item()],
            'best_conf': conf.item(),
            'best_cls': cls.item(),
            'image_id': img_path,
            'activations': [p.item() for p in activations],
            'logits': [p.item() for p in logits]
        }
        boxes.append(box_dict)

    return boxes

def run_predict(img, img_path, model, hooks, num_classes, conf_threshold=0.5, iou_threshold=0.7):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.

    Args:
        img_path (str): Path to an image file.
        model (YOLO): YOLO model object.
        hooks (list): List of hooks for the model.
        num_classes (int): Number of classes.
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
            'image_id': img_path,
            'bbox': [x0.item(), y0.item(), x1.item(), y1.item()], # xyxy
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
    """
    nms_results = yolo_ops.non_max_suppression(
        boxes_for_nms,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        nc=detect.nc)[0]
    
    # Unpack and return in bbox2result format
    class_results = [[] for _ in range(num_classes)]

    for b in range(nms_results.shape[0]):
        box = nms_results[b, :]
        x0, y0, x1, y1, conf, cls, *acts_and_logits = box
        logits = acts_and_logits[detect.nc:]
        cls_idx = int(cls.item())
        bbox = [x0.item(), y0.item(), x1.item(), y1.item()] # xyxy
        score = conf.item()
        logits = [p.item() for p in logits]  # list of floats, len == num_classes

        # Pack [x1, y1, x2, y2, score, logit_0, ..., logit_C]
        row = bbox + [score] + logits
        class_results[cls_idx].append(row)

    # Convert each class list to np.ndarray of shape (k_i, 5+num_classes)
    class_results = [np.array(cls_boxes, dtype=np.float32) if cls_boxes else np.zeros((0, 5+num_classes), dtype=np.float32)
                     for cls_boxes in class_results]
    
    """
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

def run_inference_yolo(model_path, images_folder, num_classes, confThresh=0.5, iouThresh=0.7):
    """
    Process raw YOLO output from images in a folder using captured logits.
    Collects detections per image efficiently using vectorized operations.

    Args:
        model: Trained Ultralytics YOLO model.
        images_folder (str or Path): Path to folder containing images.
        num_classes (int): Number of classes and logits.
        confThresh (float): Confidence threshold for keeping detections.
        iou_threshold (float): IOU threshold for Non-Maximum Suppression (NMS).

    Returns:
        dict: {image_name: list of detections}, each detection = [x1, y1, x2, y2, score, class_idx]
    """
    maxOneDetOneRP = True
    batch_size = 1
    images_folder = Path(images_folder)
    image_paths = sorted(list(images_folder.glob('*.jpg')))
    allResults = {}

    # Load model
    model, hooks = load_and_prepare_model(model_path)

    # Preload images (keep original path and loaded image)
    preloaded_images = [(p, cv2.imread(str(p))) for p in image_paths]

    # Main loop
    for image_path, img in tqdm.tqdm(preloaded_images, total=len(preloaded_images)):
        imgName = image_path.name
        allResults[imgName] = []
        all_detections = []

        # Predict logits using image and path
        results = run_predict(img, image_path, model, hooks, num_classes, conf_threshold=confThresh, iou_threshold=iouThresh)

        for cls_idx in range(num_classes):
            imDets = results[cls_idx]
            if len(imDets) > 0:
                logits = imDets[:, 5:5 + num_classes]
                scores = imDets[:, 4]
                mask = None
                if maxOneDetOneRP:
                    mask = np.argmax(logits, axis=1) == cls_idx
                    if np.sum(mask) > 0:
                        imDets = imDets[mask]
                        scores = scores[mask]
                    else:
                        continue

                if confThresh > 0.:
                    mask = scores >= confThresh
                    if np.sum(mask) > 0:
                        imDets = imDets[mask]
                    else:
                        continue

                all_detections.append(imDets)

        if len(all_detections) == 0:
            continue
        else:
            all_detections = np.concatenate(all_detections, axis=0)
            detections, _ = np.unique(all_detections, return_index=True, axis=0)

        allResults[imgName] = detections.tolist()

    return allResults

# Save results
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_results_json(results, save_path):
    json_str = json.dumps(results, cls=NumpyEncoder)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(f"{save_path}.json", 'w') as f:
        f.write(json_str)
    print(f"Saved to {save_path}.json")

def save_single_result_json(results):
    # Create a list to store the predictions data
    predictions = []

    for result in results:
        image_id = os.path.basename(result['image_id'])#.split('.')[0]
        # image_id = result["image_id"]
        # image_id = os.path.basename(img_path).split('.')[0]
        max_category_id = result['activations'].index(max(result['activations']))
        category_id = max_category_id
        bbox = result['bbox']
        score = max(result['activations'])
        activations = result['activations']

        prediction = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score,
            'activations': activations
        }

        predictions.append(prediction)

    # Write the predictions list to a JSON file
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

# Visualization
def annotate_image_helper(image, results, class_names):
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return image

def visualize_batch(model_path: str, image_dir: str):
    # Config
    image_folder = Path(image_dir)
    num_x, num_y = 5, 6
    images_per_batch = num_x * num_y
    num_batches = 1

    # Load model
    model = YOLO(model_path)
    class_names = model.names  # class ID -> name mapping

    # Collect image paths
    all_images = sorted(image_folder.glob('*.jpg'))
    random.shuffle(all_images)

    # Pick images for all batches
    total_images = images_per_batch * num_batches
    selected_images = all_images[:total_images]

    # Process images in batches
    for page in range(num_batches):
        batch_paths = selected_images[page * images_per_batch : (page + 1) * images_per_batch]
        fig, axs = plt.subplots(num_y, num_x, figsize=(16, 16))
        axs = axs.flatten()

        for i, img_path in enumerate(batch_paths):
            img = cv2.imread(str(img_path))
            img_rgb = img[:, :, ::-1]  # BGR to RGB
            results = model.predict(img_rgb, verbose=False)[0]
                
            annotated = annotate_image_helper(img.copy(), results, class_names)
            axs[i].imshow(annotated[:, :, ::-1])  # back to RGB for matplotlib
            
            # Get list of class labels from results
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            detected_labels = [f"{class_names[cls]} ({conf:.2f})" for cls, conf in zip(clss, confs)]

            # Create title with image name and classes
            title = img_path.name
            # if detected_labels:
            #     title += ":\n" + ", ".join(detected_labels)
            axs[i].set_title(title, fontsize=8)

            axs[i].axis('off')

            # Hide empty subplots (if any)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

        plt.tight_layout()
        plt.show()

def plot_image(img_path, results, category_mapping=None, suffix='test', show_labels=True, include_legend=True):
    """
    Display the image with bounding boxes and their corresponding class scores.

    Args:
        img_path (str): Path to the image file.
        results (list): List of dictionaries containing bounding box information.
        category_mapping:
        suffix: what to append to the original image name when saving

    Returns:
        None
    """

    img = Image.open(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for box in results:
        x0, y0, x1, y1 = map(int, box['bbox'])

        box_color = "r"  # red
        tag_color = "k"  # black
        max_score = max(box['activations'])
        max_category_id = box['activations'].index(max_score)
        category_name = max_category_id

        if category_mapping:
            max_category_name = category_mapping.get(max_category_id, "Unknown")
            category_name = max_category_name

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=box_color,
            label=f"{max_category_id}: {category_name} ({max_score:.2f})",
            facecolor='none'
        )
        ax.add_patch(rect)

        if show_labels:
            plt.text(
                x0,
                y0 - 50,
                f"{max_category_id} ({max_score:.2f})",
                fontsize="5",
                color=tag_color,
                backgroundcolor=box_color,
            )

    if include_legend:
        ax.legend(fontsize="5")

    plt.axis("off")
    # plt.savefig(f'{os.path.basename(img_path).rsplit(".", 1)[0]}_{suffix}.jpg', bbox_inches="tight", dpi=300)

# CLI
def main():
    parser = argparse.ArgumentParser(description='Extract logits for train, val, test set.')
    parser.add_argument('model_path', type=str, help='Path to YOLO model.')
    parser.add_argument('image_dir', type=str, help='Path to image directory.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes.')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='Confidence Threshold.')
    parser.add_argument('--iou_thresh', type=float, default=0.7, help='IOU Threshold.')
    args = parser.parse_args()

    print('Extracting features!')
    results = run_inference_yolo(args.model_path,
                            f'{args.image_dir}',
                            num_classes=args.num_classes,
                            confThresh=args.conf_thresh,
                            iouThresh=args.iou_thresh)
    # TODO: Setting path
    split = args.image_dir.split("/")[-1]
    save_results_json(results, f'/home/chen/TMNF/data/extracted_feat/flowDet/YOLOv8/associated/VOCDataset/extraction_{split}')


if __name__ == "__main__":
    main()
