import os
import numpy as np
from pycocotools import mask as mask_utils

def parse_mots_line(line):
    parts = line.strip().split()
    frame = int(parts[0])
    obj_id = int(parts[1])
    class_id = int(parts[2])
    img_height = int(parts[3])
    img_width = int(parts[4])
    rle = " ".join(parts[5:])
    return frame, obj_id, class_id, img_height, img_width, rle

def rle_to_bbox(rle, height, width):    
    rle_obj = {
        "size": [height, width], 
        "counts": rle.encode(encoding='UTF-8')
    }
    # binary_mask = mask_utils.decode(rle_obj)
    # rle_ = mask_utils.encode(np.asfortranarray(binary_mask))
    # bbox = mask_utils.toBbox(rle_)
    bbox = mask_utils.toBbox(rle_obj)
    if bbox.size == 0:
        return None

    # NOTE: GPT Önerdi..
    # x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    # return x_min, y_min, x_max - x_min, y_max - y_min

    # NOTE: CCA.py kodundan alındı
    x, y, w, h = bbox.astype(int) #int çevirilecek
    #x1, y1, w1, h1 = bbox
    return x, y, w, h

def convert_mots_to_motchallenge(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    results = []
    for line in lines:
        frame, obj_id, class_id, img_height, img_width, rle = parse_mots_line(line)
        
        if obj_id == 10000 and class_id == 10: #Ignored region kısmı
            conf = 0
            obj_id = 10000
        else:
            if class_id == 2:
                conf = 1
                obj_id = int(str(obj_id)[1:])
            else:
                conf = -1
                obj_id = obj_id

        bbox = rle_to_bbox(rle, img_height, img_width)
        if bbox is None:
            continue
        
        bb_left, bb_top, bb_width, bb_height = bbox
        results.append(f"{frame},{obj_id},{bb_left},{bb_top},{bb_width},{bb_height},{conf},-1,-1,-1\n")
    
    with open(output_file, 'w') as f:
        f.writelines(results)

def process_conversion(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            convert_mots_to_motchallenge(input_path, output_path)
    
    print(f"Conversion completed. Output files saved in {output_dir}")

if __name__ == "__main__":
    base_path = '/cta/users/grad4/master/datasets/MOTS/train/'
    mots_path = 'MOTS20-09/'
    input_directory = base_path + mots_path + "gt"
    output_directory = base_path + mots_path + "gt_mot"
    process_conversion(input_directory, output_directory)
