from pathlib import Path
from PIL import Image
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
import datasets.transforms as T
import json
import os

class TAODataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None, return_masks=False):
        """
        Initializes the TAO dataset.

        Args:
            img_folder (str): Directory where images are stored.
            ann_file (str): Path to the annotations JSON file.
            transforms (callable, optional): Transforms to be applied to the images and targets.
            return_masks (bool): Whether to return masks or not.
        """
        self.img_folder = img_folder
        self.return_masks = return_masks
        self.transforms = transforms
        self.data = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):
        """
        Loads annotations from a JSON file.

        Args:
            ann_file (str): Path to the annotations JSON file.

        Returns:
            list: List of annotations.
        """
        with open(ann_file, 'r') as f:
            return json.load(f)['sequences']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        image_id = sequence['seq_name']  # Assuming seq_name is used as image ID

        # Load images
        img = []
        for image_path in sequence['all_image_paths']:
            img_path = os.path.join(self.img_folder, image_path)
            img.append(self.load_image(img_path))

        # Load annotations
        annotations = self.prepare_annotations(sequence)

        # Apply transforms
        if self.transforms:
            img, annotations = self.transforms(img, annotations)

        return img, annotations

    def load_image(self, img_path):
        # Load and return the image
        return Image.open(img_path).convert("RGB")

    def prepare_annotations(self, sequence):
        # Filter only person class annotations
        annotations = []
        for frame in sequence['segmentations']:
            for track_id, segmentation in frame.items():
                annotations.append({
                    'image_id': sequence['seq_name'],
                    'segmentation': segmentation,
                    'bbox': self.get_bbox(segmentation),
                    'category_id': 0  # Assuming 'person' category ID is 0
                })
        return annotations

    def get_bbox(self, segmentation):
        # Calculate bounding box from segmentation
        # This should be replaced with the appropriate calculation for your format
        x_min = min([p[0] for p in segmentation])
        y_min = min([p[1] for p in segmentation])
        x_max = max([p[0] for p in segmentation])
        y_max = max([p[1] for p in segmentation])
        return [x_min, y_min, x_max - x_min, y_max - y_min]

def make_tao_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'Unknown image_set: {image_set}')

def build_tao_dataset(image_set, data_root):
    img_folder = Path(data_root) / "frames/train"
    ann_file = Path(data_root) / "annotations" / "instances_train.json"

    dataset = TAODataset(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=make_tao_transforms(image_set),
        return_masks=True
    )
    return dataset
