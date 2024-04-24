# 自动化-232-蒋一璠-231404010211
# coding=gbk
# -*- coding:utf-8 -*-

import json     # 导入json库,opencv,cocotools
import os
import requests
import cv2
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

#   （1）读取json文件
#   load coco data
json_path = r'E:\download\edge\TestData_coco.json'  # json文件的保存地址
coco = COCO(annotation_file=json_path)  # 用coco库导入json文件
#   获取图像索引 ids
ids = list(sorted(coco.imgs.keys()))
coco_classes = dict([(v['id'], v['name']) for k, v in coco.cats.items()])
#   读取json文件
with open(r'E:\download\edge\TestData_coco.json', 'r', encoding='utf-8') as file:

    a = file.read()
    dic_a = json.loads(a)
    formatted_json = json.dumps(dic_a, indent=4)
#   print(a)

#   （2）下载图片并命名
#   print(formatted_json)
images_data = dic_a["images"]
#   print(images_data, '\n', type(images_data), type(dic_a))  # dic_a是字典，images_data是列表
#   images_data_length = len(images_data)   # images_data长度是十
#   idl = images_data_length  # images_data长度是idl
#   print(idl)
# 创建下载的文件本地存放目录
down_path = r'E:\coco\03task\picture'
os.makedirs(down_path, exist_ok=True)

# 待下载的图片地址列表
pictures = [images_data[0]['coco_url'], images_data[1]['coco_url'],
            images_data[2]['coco_url'], images_data[3]['coco_url'],
            images_data[4]['coco_url'], images_data[5]['coco_url'],
            images_data[6]['coco_url'], images_data[7]['coco_url'],
            images_data[8]['coco_url'], images_data[9]['coco_url']
            ]


for pic_url in ids:
    pic_file = down_path + "\\" + str(pic_url) + '.jpg'
    # 下载一张图片并以指定的文件名存放到指定路径
    urlretrieve(coco.imgs[pic_url]['coco_url'], pic_file)
    print(pic_file, "done...")

#   (3)为图片724建立一个字典
annotations = dic_a['annotations']
categories = dic_a['categories']
picture_724 = {'image_ids': annotations[0]['image_id'],
               'segmentations': annotations[0]['segmentation'],
               'bboxes': annotations[0]['bbox'],
               'category_ids': annotations[0]['category_id']
               }
#   print(picture_724)

#   （4）可视化图像1000，标签，分割信息
#   提取图片1000的类别信息
catIds = coco.getCatIds(catNms=['person', 'tennis racket', 'backpack', 'handbag'])
catInfo = coco.loadCats(catIds)
print(f"catIds:{catIds}")
print(f"catcls:{catInfo}")
#   图像信息
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)
imgInfo = coco.loadImgs(imgIds[0])[0]
print(f"imgIds:{imgIds}")
print(f"img:{imgInfo}")
#   标注信息
annIds = coco.getAnnIds(imgIds=imgInfo['id'], catIds=catIds, iscrowd=None)
annsInfo = coco.loadAnns(annIds)
print(f"annIds:{annIds}")
print(f"annsInfo:{annsInfo}", '\n', type(annsInfo), type(annsInfo[0]))
#   可视化标签和图框
img_name = r"E:\coco\03task\picture\1000.jpg"  # 图片1000本地地址
img = cv2.imread(img_name)  # 用opencv库函数读取图片
for i in range(17):         # 17为标签数量
    box = coco.imgToAnns[1000][i]['bbox']
    box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]  # 转化为右下坐标
    box_color = (255, 0, 255)  # 指定颜色
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=box_color, thickness=1)  # 画矩形框
    cv2.putText(img, coco.cats[coco.imgToAnns[1000][i]['category_id']]['name'],
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (250, 0, 250), thickness=1, lineType=1)  # 标注信息
cv2.imshow('1000.jpg', img)  # 显示图像，框，标注
plt.imshow(img)
coco.showAnns(annsInfo)  # 显示分割信息
plt.show()
cv2.waitKey(0)  # 规定窗口存在时间
cv2.destroyAllWindows()

#   （5）新的json文件
dic_extract = {'annotation': [], 'categories': [], 'images': []}  # 建立一个字典
# 抽取指定图像id的分割标注信息
ann_ids = coco.getAnnIds(imgIds=[139, 724, 785, 885, 1000])
anns = coco.loadAnns(ann_ids)
print(anns)
# 抽取指定图像id的图像信息
img_extract = coco.loadImgs(ids=[139, 724, 785, 885, 1000])
print(img_extract)
dic_extract['annotation'] = anns
dic_extract['categories'] = dic_a['categories']
dic_extract['images'] = img_extract
# 写入新的json文件
with open("dic_extract", "w") as f:
    json.dump(dic_extract, f)
