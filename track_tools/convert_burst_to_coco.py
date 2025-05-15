import json, cv2
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

class BurstToCocoConverter:
    def __init__(self, burst_json_path):
        self.burst_json_path = burst_json_path
        self.coco_data = {
            ##TODO add Videos if necessary
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.category_id_map = {}
        self.annotation_id = 1
        self.image_cnt = 1

    def load_burst_data(self):
        with open(self.burst_json_path, 'r') as f:
            self.burst_data = json.load(f)

    def convert_categories(self):
        ##NOTE: Refers to categories [{'id': 1, 'name': 'pedestrian'}] in MOT
        category_counter = 1
        for category in self.burst_data['categories']:
            coco_category = {
                "id": category['id'],
                "name": category['name'],
                "supercategory": category['supercategory'],
            }
            self.coco_data["categories"].append(coco_category)
            self.category_id_map[category['id']] = category_counter
            category_counter += 1

    def convert_images_and_annotations(self):
        for seq in self.burst_data['sequences']:
            for img_index, img_path in enumerate(seq['annotated_image_paths']):
                # Add image
                #image_id = int(f"{seq['id']}{img_index}")
                ## note: bu kısımda id, frame_id kısımları karışmış olabilir..., detaylı bakılacak... 
                coco_image = {
                    "id": self.image_cnt,
                    "file_name": f"{seq['dataset']}/{seq['seq_name']}/{img_path}",
                    "frame_id": img_index + 1,
                    "prev_image_id": img_index if img_index != 0 else -1,
                    "next_image_id": img_index + 2 if img_index + 1 < len(seq['annotated_image_paths']) else -1,
                    "video_id": seq['id'],
                    "width": seq['width'],
                    "height": seq['height'],
                    ##NOTE: Below two added for possibility to handle ignored regions. Since there is no directly ignored regions are proposed.
                    "neg_category_ids": seq['neg_category_ids'],
                    "not_exhaustive_category_ids": seq['not_exhaustive_category_ids']
                }
                self.coco_data["images"].append(coco_image)
                # Add annotations for each track in the frame
                if img_index < len(seq['segmentations']):
                    self.add_annotations(seq, img_index, self.image_cnt)
                self.image_cnt += 1

    def add_annotations(self, seq, img_index, image_id):
        segmentation_data = seq['segmentations'][img_index]
        for track_id, track_info in segmentation_data.items():
            ##NOTE: segmentation_data.items() deki bazı annotasyonlar {} yani boz dict olarak geliyor. bunları coco ya eklerken ya kaldıracaz(Bu durumda o resmide datasetten silmek gerekli), Ya dacoco formatindaki halinede {} boş dict koyucaz bu durumda eğitim sırasında nasıl handle edicez ona bakmak lazım
            ##NOTE: yukarıda dediğim durumu debug console tarfında gorebilirsin 016 ve 017 boş geliyor.
            ##NOTE: AMK DATASETİ AŞIRI SORUNLU.
            category_id = seq['track_category_ids'][track_id]

            # Process RLE mask and ensure correct format
            rle = track_info['rle']
            if isinstance(rle, str):
                # Wrap in a dictionary if it's an RLE string
                rle = {"counts": rle, "size": [seq['height'], seq['width']]}
            elif isinstance(rle, dict) and "counts" in rle and "size" in rle:
                # Already properly formatted
                pass
            else:
                raise ValueError("RLE mask format is unsupported or invalid.")

            # Decode the RLE mask to a binary mask
            binary_mask = mask_util.decode(rle)

            # Calculate area and bounding box from binary mask
            area = int(binary_mask.sum())
            bbox = mask_util.toBbox(mask_util.encode(np.asfortranarray(binary_mask))).tolist()

            # Create annotation
            coco_annotation = {
                "id": self.annotation_id, ##NOTE: Buranın mantığına bakmak lazım sürekli 1 mi verilecek??? Gerek yoksa siline bilir.
                "image_id": image_id, ##NOTE. 0 yerine 1 den başlaması için. Buradaki image id teknik oalrak img_index ve o da hangi img olduğunu söylüyor. Onun içinde detracking id ler var yani direk img_id veya index i koyarsak olur.
                "category_id": self.category_id_map[1], ##NOTE: Buranın kesinlikle 805 tarafıyla uyuşması lazım. 805 i 1 e script içinde çekeceksek 1 olması aksi takdirde 805 olmalı. Ben ne geldiğine bakmadım.
                #"conf": track_info['score'], ##NOTE: Bir deetctor tarafından detectt edilmişse 1 den farklı olabiliyor. Çok düşükleri almamk lazım ama gebel olarak 0.91 den büyükler endüşük 0.85 gördüm belki bunalrın atılması gerekebilir. 
                "track_id": track_id,
                "segmentation": rle, ##NOTE: rle den binary maska çeviridm dataloaderda düzgün mü bakmak lazım
                "area": area,
                "bbox": bbox,
                "iscrowd": 0, ##NOTE: bu özellik aslında data sette ignored region varmı gosteriyor. Bunun 1 olduğu annotasyon ignored regions a tekabül ediyor. Ancak datasetin orjinalinde öyle bir özellik yok. Sadece kendi apilarında  ve evallerinde bir şekilde handle etmeyi bulmuşlar oradn bakılabilir.
                "is_gt": track_info['is_gt']
            }
            self.coco_data["annotations"].append(coco_annotation)
            self.annotation_id += 1


    def save_coco_data(self, output_json_path):
        with open(output_json_path, 'w') as f:
            json.dump(self.coco_data, f)

    def convert(self, output_json_path):
        self.load_burst_data()
        self.convert_categories()
        self.convert_images_and_annotations()
        self.save_coco_data(output_json_path)


    def display_annotations(image_id, json_path, image_dir):
        # Load the COCO-style annotations
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Initialize the COCO API for instance annotations
        coco = COCO(json_path)
        
        # Get image info and load the image
        image_info = coco.loadImgs(image_id)[0]
        image_path = f"{image_dir}/{image_info['file_name']}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        
        # Load annotations for the image
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            # Draw the bounding box
            bbox = ann['bbox']
            x, y, w, h = bbox
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none'))

            # Decode and draw the mask
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):  # Polygon format
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((len(seg) // 2, 2))
                        plt.gca().add_patch(plt.Polygon(poly, linewidth=2, edgecolor='red', facecolor='none'))
                else:  # RLE format
                    mask = mask_util.decode(ann['segmentation'])
                    plt.imshow(mask, alpha=0.5, cmap='jet')
        
        plt.show()


if __name__ == "__main__":
    burst_json_path = '/cta/users/grad4/master/datasets/tao/person/annotations/burst/v2/instances_train_burst.json'
    output_json_path = '/cta/users/grad4/master/datasets/tao/person/annotations/instances_train.json'
    converter = BurstToCocoConverter(burst_json_path)
    converter.convert(output_json_path)

    # json_path = output_json_path
    # image_dir = 'path_to_images'
    # image_id = 1  # Replace with the ID of the image you want to display

    # #converter.display_annotations(image_id, json_path, image_dir)


## NOTE: rle to mask için asağıdaki kod satırlarını eklememiz gerekiyor, anladığıma biinary_mask sadece numpy array olarak geliyor ama busrt api'den baktığım kadarıyla rle'yi mask'a su sekilde convert ediyor...
## NOTE: bu asağıdaki kod kısımlarını bizim kod'a yedirmemiz gerekiyor...
## NOTE: ana kod'a master/dataset/BURST-Banchmark kodu altından burstapi klasörü altındaki demo.py dosyasında 54. satırdan itibaren incelenerek detaylı bakılabilir. 
# def rle_ann_to_mask(rle: str, image_size: Tuple[int, int]) -> np.ndarray:
#     return mask_util.decode({
#         "size": image_size,
#         "counts": rle.encode("utf-8")
#     }).astype(bool)

# def load_masks(self, frame_indices: Optional[List[int]] = None) -> List[Dict[int, np.ndarray]]:
#         """
#         Decode RLE masks into mask images
#         :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
#         0 <= t < len(self.num_annotated_frames)
#         :return: List of dicts (one per frame). Each dict has track IDs as keys and mask images as values.
#         """
#         if frame_indices is None:
#             frame_indices = list(range(self.num_annotated_frames))
#         else:
#             assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
#                 f"invalid"

#         zero_mask = np.zeros(self.image_size, bool)
#         masks = []

#         for t in frame_indices:
#             masks_t = dict()

#             for track_id in self.track_ids:
#                 if track_id in self.segmentations[t]:
#                     masks_t[track_id] = rle_ann_to_mask(self.segmentations[t][track_id]["rle"], self.image_size)
#                 else:
#                     masks_t[track_id] = zero_mask

#             masks.append(masks_t)

#         return masks

