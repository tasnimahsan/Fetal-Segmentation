import json
import cv2
import os
import numpy as np
from glob import glob
from PIL import Image


def createMaskFromJson(json_files, output_dir, category):
    counter = 0
    for image_json in json_files:
        with open(image_json) as file:
            data = json.load(file)
#         filename = data["imagePath"].split(".")[2].split("/")[2]
        # print(filename)

        # creating a new ground truth image
        mask = np.zeros(
            (data["imageHeight"], data["imageWidth"]), dtype="uint8")
        for shape in data["shapes"]:
            mask = cv2.fillPoly(
                mask, [np.array(shape["points"], dtype=np.int32)], color=255)

        # saving the ground truth masks
        cv2.imwrite(os.path.join(
            output_dir, "{}_{}.jpg".format(counter, category)), mask)
        counter += 1


def changeFileNames(image_path, output_dir, category):
    counter = 0
    for image in image_path:
        os.rename(image, output_dir+'{}_{}.jpg'.format(counter, category))
        counter += 1


mask_output_dir = "../Image/Train/Mask"
os.makedirs(mask_output_dir, exist_ok=True)

hc_json = sorted(glob("../Data/HC/Json Mask/*.json"))
ac_json = sorted(glob("../Data/AC/Json Mask/*.json"))
fl_json = sorted(glob("../Data/FL/Json Mask/*.json"))

createMaskFromJson(hc_json, mask_output_dir, "hc")
createMaskFromJson(ac_json, mask_output_dir, "ac")
createMaskFromJson(fl_json, mask_output_dir, "fl")

raw_output_dir = "../Image/Train/Raw/"
os.makedirs(raw_output_dir, exist_ok=True)

hc_raw = sorted(glob("../Data/HC/Training/*.jpg"))
ac_raw = sorted(glob("../Data/AC/Training/*.jpg"))
fl_raw = sorted(glob("../Data/FL/Training/*.jpg"))

changeFileNames(hc_raw, raw_output_dir, "hc")
changeFileNames(ac_raw, raw_output_dir, "ac")
changeFileNames(fl_raw, raw_output_dir, "fl")

raw_output_dir = "../Image/Test/"
os.makedirs(raw_output_dir, exist_ok=True)

hc_test = sorted(glob("../Data/HC/Testing/*.jpg"))
ac_test = sorted(glob("../Data/AC/Testing/*.jpg"))
fl_test = sorted(glob("../Data/FL/Testing/*.jpg"))

changeFileNames(hc_test, raw_output_dir, "hc")
changeFileNames(ac_test, raw_output_dir, "ac")
changeFileNames(fl_test, raw_output_dir, "fl")
