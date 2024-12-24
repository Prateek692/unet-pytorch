import os
import json
import cv2
import numpy as np

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

img_folder = # Path to images folder
label_folder = # Path to labels folder

os.makedirs(label_folder, exist_ok=True)

json_file_path = # path to coco json file

with open(json_file_path, "r") as f:
    annots_content = json.load(f)
    categories = {}
    i = 0
    for c in annots_content["categories"]:
        categories.update({c['id']: [c['name'], colors[i]]})
        i += 1

    images = annots_content["images"]
    for image in images:
        image_id = image["id"]
        image_name = image["file_name"]
        img = cv2.imread(os.path.join(img_folder, image_name))

        annots = annots_content["annotations"]

        label_file = np.zeros((1216,1936,4), dtype=np.uint8)

        for annot in annots:
            if annot["image_id"] == image_id:
                seg = np.array(annot["segmentation"][0]).reshape(-1, 2).astype(np.int32)
                cv2.drawContours(img, [seg], -1, categories[annot['category_id']][1], 3)

                cv2.drawContours(label_file, [seg], -1, annot['category_id'], 2)
                cv2.fillPoly(label_file, [seg], annot['category_id'])
                print(label_file.shape)
                
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        
        cv2.namedWindow("label", cv2.WINDOW_NORMAL)
        cv2.imshow("label", label_file*63)
        print(os.path.join(label_folder, image_name))
        cv2.imwrite(os.path.join(label_folder, image_name), label_file)
        cv2.waitKey(1)