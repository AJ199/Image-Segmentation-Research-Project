# dataset_register.py
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def register_my_dataset(name, json_anno, img_dir):
    """
    name: str - dataset name (e.g., "city_train")
    json_anno: path to COCO-format annotations json
    img_dir: path to images directory
    """
    register_coco_instances(name, {}, json_anno, img_dir)
    meta = MetadataCatalog.get(name)
    return meta
