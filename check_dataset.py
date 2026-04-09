# check_dataset.py  —  Verify captions from all 3 sources
#
# Prints first 2 image paths + captions from:
#  1. Flickr8k
#  2. COCO 2014
#  3. COCO 2017
#
# Run: python check_dataset.py
import os
import json
from config import cfg

DATA_DIR = cfg.data_dir


def check_flickr8k():
    print("\n" + "=" * 55)
    print("  SOURCE 1 — Flickr8k")
    print("=" * 55)

    captions_file = os.path.join(DATA_DIR, "captions.txt")
    image_dir = os.path.join(DATA_DIR, "Images")

    if not os.path.exists(captions_file):
        print("  ❌ captions.txt not found")
        return

    count = 0
    with open(captions_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line_num == 0:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            image_name, caption = parts
            image_path = os.path.join(image_dir, image_name.strip())
            if os.path.exists(image_path):
                print(f"\n  Image:   {image_name.strip()}")
                print(f"  Caption: {caption.strip()}")
                count += 1
                if count == 2:
                    break

    if count == 0:
        print("  ❌ No valid pairs found")
    else:
        print(f"\n  ✅ Flickr8k is working")


def check_coco(json_path, image_dir, label):
    print("\n" + "=" * 55)
    print(f"  SOURCE {label}")
    print("=" * 55)

    if not os.path.exists(json_path):
        print(f"  ❌ JSON not found: {json_path}")
        return
    if not os.path.exists(image_dir):
        print(f"  ❌ Image dir not found: {image_dir}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    count = 0
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"].strip()

        if image_id not in id_to_filename:
            continue

        filename = id_to_filename[image_id]
        image_path = os.path.join(image_dir, filename)

        if os.path.exists(image_path):
            print(f"\n  Image:   {filename}")
            print(f"  Caption: {caption}")
            count += 1
            if count == 2:
                break

    if count == 0:
        print("  ❌ No valid pairs found")
    else:
        print(f"\n  ✅ {label} is working")


if __name__ == "__main__":
    print("\n🔍 Checking all 3 dataset sources...\n")

    # Flickr8k
    check_flickr8k()

    # COCO 2014
    check_coco(
        json_path=os.path.join(
            DATA_DIR, "annotations_trainval2014", "captions_train2014.json"
        ),
        image_dir=os.path.join(DATA_DIR, "train2014"),
        label="2 — COCO 2014 train",
    )

    # COCO 2017
    check_coco(
        json_path=os.path.join(
            DATA_DIR, "annotations_trainval2017", "captions_val2017.json"
        ),
        image_dir=os.path.join(DATA_DIR, "val2017"),
        label="3 — COCO 2017 val",
    )

    print("\n" + "=" * 55)
    print("  Done! All sources checked.")
    print("=" * 55 + "\n")
