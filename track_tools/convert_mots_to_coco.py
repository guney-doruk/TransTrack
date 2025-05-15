# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Generates COCO data and annotation structure from MOTChallenge data.
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import numpy as np
import pycocotools.mask as rletools
import skimage.io as io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou

class SegmentedObject:
    """
    Helper class for segmentation objects.
    """
    def __init__(self, mask: dict, class_id: int, track_id: int) -> None:
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def load_mots_gt(path: str) -> dict:
    """Load MOTS ground truth from path."""
    objects_per_frame = {}
    track_ids_per_frame = {}  # Check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # Check that no frame contains overlapping masks

    with open(path, "r") as gt_file:
        for line in gt_file:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {
                'size': [int(fields[3]), int(fields[4])],
                'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([
                    combined_mask_per_frame[frame], mask],
                    intersect=True)):
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask],
                    intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame

MOTS_ROOT = '/cta/users/grad4/master/datasets/MOTS'
VIS_THRESHOLD = 0.25

MOT_15_SEQS_INFO = {
    'ETH-Bahnhof': {'img_width': 640, 'img_height': 480, 'seq_length': 1000},
    'ETH-Sunnyday': {'img_width': 640, 'img_height': 480, 'seq_length': 354},
    'KITTI-13': {'img_width': 1242, 'img_height': 375, 'seq_length': 340},
    'KITTI-17': {'img_width': 1224, 'img_height': 370, 'seq_length': 145},
    'PETS09-S2L1': {'img_width': 768, 'img_height': 576, 'seq_length': 795},
    'TUD-Campus': {'img_width': 640, 'img_height': 480, 'seq_length': 71},
    'TUD-Stadtmitte': {'img_width': 640, 'img_height': 480, 'seq_length': 179},}


def generate_coco_from_mot(split_name='train', seqs_names=None,
                           root_split='train', mots=False, mots_vis=False,
                           frame_range=None, data_root='/cta/users/grad4/master/datasets/MOTS'):
    """
    Generates COCO data from MOT.
    """
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    if mots:
        data_root = MOTS_ROOT
    root_split_path = os.path.join(data_root, root_split)
    root_split_mots_path = os.path.join(MOTS_ROOT, root_split)
    coco_dir = os.path.join(data_root, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "person",
                                  "name": "person",
                                  "id": 1}]
    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0

    seqs = sorted(os.listdir(root_split_path))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, seqs)

    video_cnt = 0 #Added for compatibility with TransTrack(Not in Orijinal code)
    for seq in seqs:
        # CONFIG FILE
        video_cnt += 1
        config = configparser.ConfigParser()
        config_file = os.path.join(root_split_path, seq, 'seqinfo.ini')

        if os.path.isfile(config_file):
            config.read(config_file)
            img_width = int(config['Sequence']['imWidth'])
            img_height = int(config['Sequence']['imHeight'])
            seq_length = int(config['Sequence']['seqLength'])
        else:
            img_width = MOT_15_SEQS_INFO[seq]['img_width']
            img_height = MOT_15_SEQS_INFO[seq]['img_height']
            seq_length = MOT_15_SEQS_INFO[seq]['seq_length']

        seg_list_dir = sorted(os.listdir(os.path.join(root_split_path, seq, 'img1')))
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)

        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id + 1 # +1 Added for compatibility with TransTrack(Not in Orijinal code)

            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id + 1, # +1 Added for compatibility with TransTrack(Not in Orijinal code)
                                          "frame_id": i + 1, # +1 Added for compatibility with TransTrack(Not in Orijinal code)
                                          "video_id": video_cnt,#Added for compatibility with TransTrack(Not in Orijinal code)
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            img_id += 1

            os.symlink(os.path.join(os.getcwd(), root_split_path, seq, 'img1', img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # GT
    annotation_id = 1 #Original 0 but since we start everthing from 1 we changed this also.
    img_file_name_to_id = {
        img_dict['file_name']: img_dict['id']
        for img_dict in annotations['images']}
    for seq in seqs:
        # GT FILE
        gt_file_path = os.path.join(root_split_path, seq, 'gt', 'gt.txt')
        if mots:
            gt_file_path = os.path.join(
                root_split_mots_path,
                seq.replace('MOT17', 'MOTS20'),
                'gt',
                'gt.txt')
        if not os.path.isfile(gt_file_path):
            continue

        seq_annotations = []
        if mots:
            mask_objects_per_frame = load_mots_gt(gt_file_path)
            for frame_id, mask_objects in mask_objects_per_frame.items():
                for mask_object in mask_objects:
                    # class_id = 1 is car
                    # class_id = 2 is person
                    # class_id = 10 IGNORE
                    if mask_object.class_id == 1:
                        continue

                    bbox = rletools.toBbox(mask_object.mask)
                    bbox = [int(c) for c in bbox]
                    area = bbox[2] * bbox[3]
                    image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.jpg", None)
                    if image_id is None:
                        continue

                    segmentation = {
                        'size': mask_object.mask['size'],
                        'counts': mask_object.mask['counts'].decode(encoding='UTF-8')}

                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": image_id,
                        "segmentation": segmentation,
                        "ignore": mask_object.class_id == 10,
                        "visibility": 1.0,
                        "area": area,
                        "iscrowd": 0,
                        "seq": seq,
                        "category_id": annotations['categories'][0]['id'] if mask_object.class_id != 10 else -1,
                        "track_id": mask_object.track_id}

                    seq_annotations.append(annotation)
                    annotation_id += 1

            annotations['annotations'].extend(seq_annotations)
        else:

            seq_annotations_per_frame = {}
            with open(gt_file_path, "r") as gt_file:
                reader = csv.reader(gt_file, delimiter=' ' if mots else ',')

                for row in reader:
                    if int(row[6]) == 1 and (seq in MOT_15_SEQS_INFO or int(row[7]) == 1):
                        bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                        bbox = [int(c) for c in bbox]

                        area = bbox[2] * bbox[3]
                        visibility = float(row[8])
                        frame_id = int(row[0])
                        image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.jpg", None)
                        if image_id is None:
                            continue
                        track_id = int(row[1])

                        annotation = {
                            "id": annotation_id,
                            "bbox": bbox,
                            "image_id": image_id,
                            "segmentation": [],
                            "ignore": 0 if visibility > VIS_THRESHOLD else 1,
                            "visibility": visibility,
                            "area": area,
                            "iscrowd": 0,
                            "seq": seq,
                            "category_id": annotations['categories'][0]['id'],
                            "track_id": track_id}

                        seq_annotations.append(annotation)
                        if frame_id not in seq_annotations_per_frame:
                            seq_annotations_per_frame[frame_id] = []
                        seq_annotations_per_frame[frame_id].append(annotation)

                        annotation_id += 1

            annotations['annotations'].extend(seq_annotations)

            #change ignore based on MOTS mask
            if mots_vis:
                gt_file_mots = os.path.join(
                    root_split_mots_path,
                    seq.replace('MOT17', 'MOTS20'),
                    'gt',
                    'gt.txt')
                if os.path.isfile(gt_file_mots):
                    mask_objects_per_frame = load_mots_gt(gt_file_mots)

                    for frame_id, frame_annotations in seq_annotations_per_frame.items():
                        mask_objects = mask_objects_per_frame[frame_id]
                        mask_object_bboxes = [rletools.toBbox(obj.mask) for obj in mask_objects]
                        mask_object_bboxes = torch.tensor(mask_object_bboxes).float()

                        frame_boxes = [a['bbox'] for a in frame_annotations]
                        frame_boxes = torch.tensor(frame_boxes).float()

                        # x,y,w,h --> x,y,x,y
                        frame_boxes[:, 2:] += frame_boxes[:, :2]
                        mask_object_bboxes[:, 2:] += mask_object_bboxes[:, :2]

                        mask_iou = box_iou(mask_object_bboxes, frame_boxes)

                        mask_indices, frame_indices = linear_sum_assignment(-mask_iou)
                        for m_i, f_i in zip(mask_indices, frame_indices): 
                            if mask_iou[m_i, f_i] < 0.5:
                                continue

                            if not frame_annotations[f_i]['visibility']:
                                frame_annotations[f_i]['ignore'] = 0

    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


def check_coco_from_mot(coco_dir='data/MOT17/mot17_train_coco', annotation_file='data/MOT17/annotations/mot17_train_coco.json', img_id=None):
    """
    Visualize generated COCO data. Only used for debugging.
    """
    # coco_dir = os.path.join(data_root, split)
    # annotation_file = os.path.join(coco_dir, 'annotations.json')

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    if img_id == None:
        img_ids = coco.getImgIds(catIds=cat_ids)
        index = np.random.randint(0, len(img_ids))
        img_id = img_ids[index]
    img = coco.loadImgs(img_id)[0]

    i = io.imread(os.path.join(coco_dir, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig('annotations.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO from MOT.')
    parser.add_argument('--frame_split', action='store_true')
    args = parser.parse_args()

    if args.frame_split:
        ##TODO: Fill
        pass

    else: 
        #
        # MOTS20
        #

        # TRAIN SET FULL
        # generate_coco_from_mot(
        #     'mots20_train_coco_full',
        #     seqs_names=['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11'],
        #     mots=True)

        # # # TRAIN SPLITS
        # # for i in range(4):
        # #     train_seqs = ['MOTS20-05', 'MOTS20-02', 'MOTS20-09', 'MOTS20-11']
        # #     val_seqs = train_seqs.pop(i)

        # #     generate_coco_from_mot(
        # #         f'mots20_train_{i + 1}_coco',
        # #         seqs_names=train_seqs, mots=True)
        # #     generate_coco_from_mot(
        # #         f'mots20_val_{i + 1}_coco',
        # #         seqs_names=val_seqs, mots=True)
        
        # #TRAIN SET with VAL - TRAIN PART #NOTE: 1 video for VAL
        # generate_coco_from_mot(
        #     'mots20_train_coco_09_divided',
        #     seqs_names=['MOTS20-02', 'MOTS20-05', 'MOTS20-11'],
        #     mots=True)
        
        # #TRAIN SET with VAL - VAL PART
        # generate_coco_from_mot(
        #     'mots20_val_coco_09_divided',
        #     seqs_names=['MOTS20-09'],
        #     mots=True)

        #TRAIN SET with VAL - TRAIN PART #NOTE: Half frame seperated
        generate_coco_from_mot(
            'mots20_train_coco_half',
            seqs_names=['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11'],
            mots=True,
            frame_range={'start': 0, 'end': 0.5})
        
        # #TRAIN SET with VAL - VAL PART
        generate_coco_from_mot(
            'mots20_val_coco_half',
            seqs_names=['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11'],
            mots=True,
            frame_range={'start': 0.5, 'end': 1})

        # generate_coco_from_mot(
        #      'mots20_val_coco_05_divided',
        #      seqs_names=['MOTS20-05'],
        #      mots=True)