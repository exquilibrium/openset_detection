# Modified from: https://github.com/dimitymiller/openset_detection

import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from torchvision.ops import nms  # alternative if using TorchVision

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
    parser.add_argument('--dataset', default = 'custom', help='custom, voc or coco')
    parser.add_argument('--subset', default = None, help='train, val, test, testOOD (Maha++)')
    parser.add_argument('--dir', default = None, help='directory of object detector weights')
    parser.add_argument('--checkpoint', default = 'latest.pth', help='what is the name of the object detector weights')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

args = parse_args()


#load the config file for the model that will also return logits
if args.dataset == 'custom':
    suffix = args.saveNm[len("frcnn_GMMDet_Voc_"):] # xml, xml_10c, ardea10
    args.config = f'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712OS_wLogits_{suffix}.py' ### <<<<<<<<<<---------- hardcoded path---------->>>>>>>>>>
elif args.dataset == 'voc':
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712OS_wLogits.py'
else:
    args.config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoOS_wLogits.py'
    
###################################################################################################
##############Setup Config file ###################################################################
cfg = Config.fromfile(args.config)

# import modules from string list.
if cfg.get('custom_imports', None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg['custom_imports'])
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

# in case the test dataset is concatenated
if isinstance(cfg.data.testOS, dict):
    cfg.data.testOS.test_mode = True
elif isinstance(cfg.data.testOS, list):
    for ds_cfg in cfg.data.testOS:
        ds_cfg.test_mode = True

distributed = False

samples_per_gpu = cfg.data.testOS.pop('samples_per_gpu', 1)
if samples_per_gpu > 1:
    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    cfg.data.testOS.pipeline = replace_ImageToTensor(cfg.data.testOS.pipeline)

###################################################################################################
###############Load Dataset########################################################################
print("Building datasets")
if args.dataset == 'custom':
    num_classes_dict = {
        'xml' : 15,
        'lru1' : 3,
        'lru1_drone' : 2,
        'lru1_lander' : 2,
        'lru1_lru2' : 2,
        'ardea10' : 3,
        'ardea10_lander' : 2,
        'ardea10_lru1' : 2,
        'ardea10_lru2' : 2,
    }
    num_classes = num_classes_dict[suffix] # CS classes
    if args.subset == 'train':
        dataset = build_dataset(cfg.data.trainCS)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    elif args.subset == 'testOOD':
        dataset = build_dataset(cfg.data.testOOD)
    else:
        print('That subset is not implemented.')
        exit()
elif args.dataset == 'voc':
    num_classes = 15
    if args.subset == 'train12':
        dataset = build_dataset(cfg.data.trainCS12)
    elif args.subset == 'train07':
        dataset = build_dataset(cfg.data.trainCS07)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    else:
        print('That subset is not implemented.')
        exit()
else:
    if args.subset == 'train':
        dataset = build_dataset(cfg.data.trainCS)
    elif args.subset == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.subset == 'test':
        dataset = build_dataset(cfg.data.testOS)
    else:
        print('That subset is not implemented.')
        exit()

    if args.dataset == 'coco':
        num_classes = 50
    else:
        #for the full version of coco used to fit GMMs in the iCUB experiments
        num_classes = 80


data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)


###################################################################################################
###############Build model ########################################################################
print("Building model")

# build the model and load checkpoint
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
method_list = [func for func in dir(model) if callable(getattr(model, func))]
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, '{}/{}/{}'.format(BASE_WEIGHTS_FOLDER, args.dir, args.checkpoint), map_location='cpu')

if 'CLASSES' in checkpoint['meta']:
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
model.eval()


########################################################################################################
########################## TESTING DATA  ###############################################################
########################################################################################################
print(f"Testing {args.subset} data")
num_images = len(data_loader.dataset)

score_threshold = 0.2 # only detections with a max softmax above this score are considered valid
total = 0
allResults = {}
# Mahalanobis++
feature_id_train = []
feature_id_train_logits = []
train_labels = []
feature_id_val = []
feature_id_val_logits = []
feature_ood = []
feature_ood_logits = []
debug_print = False
for i, data in enumerate(tqdm.tqdm(data_loader, total = num_images)):   
    imName = data_loader.dataset.data_infos[i]['filename']
    
    allResults[imName] = []

    total += 1
    all_detections = None
    all_scores = []
    
    with torch.no_grad():
        result = model(return_loss = False, rescale=True, **data)[0]

        # Mahalanobis++: extract post-NMS RoI features and metadata
        feats = model.module.roi_head.saved_roi_feats                # [N_all, C, H, W]
        labels = model.module.roi_head.saved_pred_labels             # [N_all]
        scores = model.module.roi_head.saved_pred_scores             # [N_all]
        rois = model.module.roi_head.saved_rois                      # [N_all, 5]
        logits = model.module.roi_head.saved_pred_logits             # [N_all, num_classes]

        if debug_print:
            print(f"{imName}")
            print(f"Total feats shape: {feats.shape}") # [1000, 256, 7, 7]
            print(f"Total labels shape: {labels.shape}")

        # --- Mahalanobis++: extract ID train val features from GT-matched RoIs and confident TP detections ---
        if args.subset in ['train', 'val']:
            # Filter detections by score
            keep_score = scores > 0.5
            if keep_score.sum() == 0:
                continue

            filtered_feats  = feats[keep_score]               # [N_filtered, C, 7, 7], C = 256
            filtered_boxes  = rois[keep_score, 1:5]           # [N_filtered, 4]
            filtered_labels = labels[keep_score]              # [1,N] last label is background

            # Load ground-truth boxes and labels
            gtData = data_loader.dataset.get_ann_info(i)
            gt_boxes = torch.tensor(gtData['bboxes'], dtype=torch.float32)    # [M, 4]
            gt_labels = torch.tensor(gtData['labels'], dtype=torch.int64)     # [M]
            if gt_boxes.numel() == 0:
                continue  # Skip if no GT boxes

            # Match each GT box to the best RoI via IoU
            img_meta = data['img_metas'][0].data[0][0]  # Rescale boxes
            scale_factor = img_meta['scale_factor']
            gt_boxes = gt_boxes * torch.tensor(scale_factor, dtype=torch.float32)

            from mmdet.core.bbox.iou_calculators import bbox_overlaps
            ious = bbox_overlaps(gt_boxes, filtered_boxes)     # [M, N]

            # For each GT box, find best-matching RoI
            matched_feats = []
            matched_labels = []
            matched_logits = []
            
            for gt_idx in range(len(gt_boxes)):
                gt_label = gt_labels[gt_idx]
                iou_row = ious[gt_idx]                            # IoUs to all preds
                match_mask = (iou_row >= 0.7) & (filtered_labels == gt_label)

                if match_mask.sum() == 0:
                    continue  # no valid matching detection

                best_idx = iou_row[match_mask].argmax()
                selected_feat = filtered_feats[match_mask][best_idx]  # [C, 7, 7]
                selected_logit = logits[keep_score][match_mask][best_idx]  # Exclude background logit

                matched_feats.append(selected_feat)
                matched_labels.append(gt_label)
                matched_logits.append(selected_logit)

            if len(matched_feats) == 0:
                continue

            pooled_feats = torch.stack(matched_feats).mean(dim=[2, 3])  # [K, C]
            pooled_logits = torch.stack(matched_logits)  # shape: [K, num_classes - 1]

            if args.subset == 'train':
                feature_id_train.append(pooled_feats.cpu().numpy())
                feature_id_train_logits.append(pooled_logits.cpu().numpy())
                #print(np.shape(feature_id_train[0]))
                train_labels.append(torch.tensor(matched_labels).cpu().numpy())
            else: # val
                feature_id_val.append(pooled_feats.cpu().numpy())
                feature_id_val_logits.append(pooled_logits.cpu().numpy())

        # --- Mahalanobis++: extract OOD features from high-confidence test boxes ---
        elif args.subset == 'testOOD':
            # Filter by score
            keep_score = scores > 0.5
            if keep_score.sum() == 0:
                continue

            filtered_feats = feats[keep_score]            # [K, C, 7, 7]
            filtered_boxes = rois[keep_score, 1:5]        # [K, 4]
            filtered_scores = scores[keep_score]          # [K]

            # Apply NMS (IoU threshold 0.5)
            keep_inds = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)

            # Keep only NMS-surviving features
            kept_feats = filtered_feats[keep_inds]              # [M, C, 7, 7]
            pooled_feats = kept_feats.mean(dim=[2, 3])          # [M, C]
            kept_logits = logits[keep_score][keep_inds]         # [M, num_classes]

            # Append to global list
            feature_ood.append(pooled_feats.cpu().numpy())
            feature_ood_logits.append(kept_logits.cpu().numpy())
    
    #collect results from each class and concatenate into a list of all the results
    for j in range(np.shape(result)[0]):
        dets = result[j]

        if len(dets) == 0:
            continue

        bboxes = dets[:, :4]
        dists = dets[:, 5:-1]  # Excludes the last column (background logit)
        scores = dets[:, 4]
        scoresT = np.expand_dims(scores, axis=1)

        #winning class must be class j for this detection to be considered valid
        mask = np.argmax(dists, axis = 1)==j

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

# Save raw detection results (existing logic)
if args.subset in ['train', 'val', 'test']:
    jsonRes = json.dumps(allResults)
    save_dir = f'{BASE_RESULTS_FOLDER}/FRCNN/raw/{args.dataset}/{args.subset}'
    os.makedirs(save_dir, exist_ok=True) #check folders exist, if not, create it

    with open(f'{save_dir}/{args.saveNm}.json', 'w') as f:
        f.write(jsonRes)

# === Mahalanobis++ feature saving ===
if args.subset in ['train', 'val', 'testOOD']:
    save_dir = f'{BASE_RESULTS_FOLDER}/FRCNN/mahalanobis/{args.dataset}/{args.subset}'
    os.makedirs(save_dir, exist_ok=True)

    if args.subset == 'train':
        print(f"Mahalanobis-Train: {len(feature_id_train)}")
        print(f"Mahalanobis-TrainLabels: {len(train_labels)}")
        np.save(os.path.join(save_dir, f'{args.saveNm}_feature_id_train.npy'), np.concatenate(feature_id_train, axis=0))
        np.save(os.path.join(save_dir, f'{args.saveNm}_feature_id_train_logits.npy'), np.concatenate(feature_id_train_logits, axis=0))
        np.save(os.path.join(save_dir, f'{args.saveNm}_train_labels.npy'), np.concatenate(train_labels, axis=0))
    elif args.subset == 'val':
        print(f"Mahalanobis-Val: {len(feature_id_val)}")
        np.save(os.path.join(save_dir, f'{args.saveNm}_feature_id_val.npy'), np.concatenate(feature_id_val, axis=0))
        np.save(os.path.join(save_dir, f'{args.saveNm}_feature_id_val_logits.npy'), np.concatenate(feature_id_val_logits, axis=0))
    elif args.subset == 'testOOD':
        print(f"Mahalanobis-OOD: {len(feature_ood)}")
        np.save(os.path.join(save_dir, f'{args.saveNm}_feature_ood.npy'), np.concatenate(feature_ood, axis=0))
        np.save(os.path.join(save_dir, f'{args.saveNm}_feature_ood_logits.npy'), np.concatenate(feature_ood_logits, axis=0))


