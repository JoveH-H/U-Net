import os
from PIL import Image

# 原图和标签图片地址
resources_path = "./resources/Oxford-IIIT Pet"
origin_images_path = resources_path + "/images"
img_name_list = os.listdir(origin_images_path)

for img_name in img_name_list:
    if img_name[-3:] == "jpg":
        tp = Image.open(origin_images_path + '/' + img_name)
        tp.save(origin_images_path + '/' + img_name[:-3] + 'png')
        os.remove(origin_images_path + '/' + img_name)
