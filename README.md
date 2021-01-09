# loader 
安装
-------
    COCO API：pycocotools 

文件说明
-------
    2coco.py读取原json文件，转化为COCO格式，生成train.json文件
    dataloader.py读取train.json，调用PandaDataset类，加载类别、图片、标注
