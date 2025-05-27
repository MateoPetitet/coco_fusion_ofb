"""
Created on Tue May 13 10:20:46 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
# -*- coding: utf-8 -*-
import os
import json
import random
import shutil
from PIL import Image

def load_jsons(json_paths):
    return [json.load(open(path, 'r')) for path in json_paths]

def get_unique_filename(existing_names, original_name):
    """Renvoie un nom de fichier unique si doublon"""
    if original_name not in existing_names:
        existing_names.add(original_name)
        return original_name

    base, ext = os.path.splitext(original_name)
    i = 1
    new_name = f"{base}_{i}{ext}"
    while new_name in existing_names:
        i += 1
        new_name = f"{base}_{i}{ext}"
    existing_names.add(new_name)
    return new_name

def merge_datasets(datasets):
    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    categories_set = set()
    category_id_map = {}
    filename_set = set()

    for data in datasets:
        # === Catégories ===
        for cat in data["categories"]:
            if cat["name"] not in categories_set:
                categories_set.add(cat["name"])
                new_id = len(merged["categories"]) + 1
                category_id_map[cat["id"]] = new_id
                cat["id"] = new_id
                merged["categories"].append(cat)
            else:
                category_id_map[cat["id"]] = next(
                    c["id"] for c in merged["categories"] if c["name"] == cat["name"]
                )

        id_map = {}

        for img in data["images"]:
            original_id = img["id"]
            original_name = img["file_name"]
            anns_for_img = [ann for ann in data["annotations"] if ann["image_id"] == original_id]

            # Ne garder que les annotations valides
            valid_anns = []
            for ann in anns_for_img:
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4 and all(v is not None for v in bbox):
                    valid_anns.append(ann)

            if len(valid_anns) == 0:
                print(f"🛑 Image ignorée : {original_name} (aucune annotation valide)")
                continue

            new_img_id = image_id_offset
            unique_name = get_unique_filename(filename_set, original_name)
            id_map[original_id] = new_img_id

            merged["images"].append({
                "id": new_img_id,
                "file_name": unique_name,
                "width": img["width"],
                "height": img["height"]
            })

            for ann in valid_anns:
                merged["annotations"].append({
                    "id": annotation_id_offset,
                    "image_id": new_img_id,
                    "category_id": category_id_map[ann["category_id"]],
                    "bbox": ann["bbox"],
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": ann.get("iscrowd", 0)
                })
                annotation_id_offset += 1

            image_id_offset += 1

    return merged

def add_empty_images(merged_data, empty_folder):
    start_id = max((img["id"] for img in merged_data["images"]), default=0) + 1
    filename_set = {img["file_name"] for img in merged_data["images"]}
    i = 0

    for filename in os.listdir(empty_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(empty_folder, filename)
            try:
                with Image.open(path) as im:
                    width, height = im.size
                unique_name = get_unique_filename(filename_set, filename)
                merged_data["images"].append({
                    "id": start_id + i,
                    "file_name": unique_name,
                    "width": width,
                    "height": height
                })
                i += 1
            except Exception as e:
                print(f"⚠️ Erreur image vide {filename}: {e}")
    return merged_data

def split_dataset(merged_data, split=(0.7, 0.2, 0.1)):
    images = merged_data["images"]
    random.shuffle(images)

    n = len(images)
    train_end = int(split[0] * n)
    val_end = train_end + int(split[1] * n)

    split_images = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    result = {}
    for split_name, imgs in split_images.items():
        img_ids = {img["id"] for img in imgs}
        anns = [ann for ann in merged_data["annotations"] if ann["image_id"] in img_ids]
        result[split_name] = {
            "images": imgs,
            "annotations": anns,
            "categories": merged_data["categories"]
        }
    return result

def save_splits_with_images(split_data, output_dir, all_image_dirs):
    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in split_data.items():
        split_folder = os.path.join(output_dir, split_name)
        images_folder = os.path.join(split_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        for img in data["images"]:
            found = False
            for img_dir in all_image_dirs:
                candidate = os.path.join(img_dir, img["file_name"])
                if os.path.exists(candidate):
                    shutil.copy2(candidate, os.path.join(images_folder, img["file_name"]))
                    found = True
                    break
            if not found:
                print(f"⚠️ Image introuvable : {img['file_name']}")

        ann_file = os.path.join(split_folder, f"annotations_{split_name}.json")
        with open(ann_file, "w") as f:
            json.dump(data, f)
        print(f"✅ {split_name} enregistré dans {split_folder}")

def main():
    # === À PERSONNALISER ===
    coco_json_paths = [
        "/home/mateo/ssd_bis/labelme_axelle_2/dataset.json",
    ]
    image_folders = [
        "/home/mateo/ssd_bis/labelme_axelle_2/images",        # images annotées
        "/home/mateo/ssd_bis/labelme_axelle_2/vides"   # images sans annotations
    ]
    output_dir = "//home/mateo/ssd_bis/labelme_axelle_2/axelle_2_ok"
    split_ratio = (0.7, 0.2, 0.1)

    # === Pipeline ===
    datasets = load_jsons(coco_json_paths)
    merged = merge_datasets(datasets)
    merged = add_empty_images(merged, image_folders[1])
    splits = split_dataset(merged, split_ratio)
    save_splits_with_images(splits, output_dir, image_folders)

if __name__ == "__main__":
    main()
