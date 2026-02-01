# extract_symbols.py
import os
import json
import numpy as np
from PIL import Image
from skimage import measure

def mask_to_bbox_and_stats(mask_arr):
    # mask_arr: 二值 numpy array (H,W)
    labels = measure.label(mask_arr, connectivity=1)
    props = measure.regionprops(labels)
    result = []
    for p in props:
        y0, x0, y1, x1 = p.bbox
        w = x1 - x0
        h = y1 - y0
        area = p.area
        result.append({
            "bbox": [int(x0), int(y0), int(w), int(h)],
            "area": int(area),
            "centroid": [float(p.centroid[1]), float(p.centroid[0])],
            "major_axis_length": float(getattr(p, "major_axis_length", 0)),
            "minor_axis_length": float(getattr(p, "minor_axis_length", 0))
        })
    return result

def simple_caption_from_mask(mask_arr):
    if mask_arr.sum() == 0:
        return "No detectable lesion."
    stats = mask_to_bbox_and_stats(mask_arr)
    if len(stats) == 1:
        s = stats[0]
        return f"Single hypoechoic lesion located at centroid {s['centroid']}, approximate area {s['area']} pixels."
    else:
        return f"{len(stats)} lesions detected; largest area ~{max(s['area'] for s in stats)} pixels."

def process_dataset(img_dir, mask_dir, out_json):
    samples = []
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
        img_path = os.path.join(img_dir, fn)
        mask_path = os.path.join(mask_dir, fn)  # assume same name
        img = Image.open(img_path).convert("L")
        if not os.path.exists(mask_path):
            print("missing mask:", mask_path)
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        caption = simple_caption_from_mask(mask)
        bboxes = mask_to_bbox_and_stats(mask)
        samples.append({
            "image_id": fn,
            "image_path": img_path,
            "mask_path": mask_path,
            "caption": caption,
            "bboxes": bboxes
        })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

if __name__ == "__main__":
    img_path = "./data/BUSI/test_val/images"
    mask_path = "./data/BUSI/test_val/masks"
    json_path = "./data/BUSI/symbols_test.json"
    # 修改为你的路径
    process_dataset(img_path, mask_path, json_path)
