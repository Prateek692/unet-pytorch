import json
import os

root_folder = # path to root folder

coco_annotations = {"info": {"description": "my-project-name"},
                    "images": [], "annotations": [],
                    "categories": [{"id": 1, "name": "Flange_Laser_Pixel"},
                                   {"id": 2, "name": "Tread_Laser_Pixel"},
                                   {"id": 3, "name": "Extra_Tread_Point"},
                                   {"id": 4, "name": "Flange_Wall_Point"},
                                   {"id": 5, "name": "Tread_Wall_Point"}]}
image_id = 0
annotation_id = 0

images_list = []
annotations_list = []

# for sub_folder in os.listdir(ip_folder):
train_folder = os.path.join(root_folder, "_annotations.coco.json")
# for file in os.listdir(train_folder):
#     if file.endswith(".json"):
f = open(train_folder)
annot_file = json.load(f)
img_map = {}
for img in annot_file["images"]:
    img_name = img["file_name"].split("_bmp")[0] + ".bmp"
    img_tag = {"id": image_id,
                "width": 1936,
                "height": 1216,
                "file_name": img_name}
    img_map.update({img["id"]: image_id})

    images_list.append(img_tag.copy())
    img_tag.clear()
    image_id += 1

for ann in annot_file["annotations"]:
    if ann["category_id"] == 1:
        cat_id = 3
    elif ann["category_id"] == 2:
        cat_id = 1
    elif ann["category_id"] == 3:
        cat_id = 4
    elif ann["category_id"] == 4:
        cat_id = 2
    elif ann["category_id"] == 5:
        cat_id = 5
    ann_tag = {"id": annotation_id,
                "iscrowd": 0,
                "image_id": img_map[ann["image_id"]],
                "category_id": cat_id,
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann["area"]}
    annotations_list.append(ann_tag.copy())
    ann_tag.clear()
    annotation_id += 1

    coco_annotations["images"] += images_list.copy()
    images_list.clear()
    coco_annotations["annotations"] += annotations_list.copy()
    annotations_list.clear()

with open(os.path.join(root_folder, "coco_annotation_fixed.json"), "w") as outfile:
    json.dump(coco_annotations, outfile)