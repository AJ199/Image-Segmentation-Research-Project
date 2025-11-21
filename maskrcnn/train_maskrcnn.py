# train_maskrcnn.py
import os
import argparse
from detectron2.engine import DefaultTrainer, default_setup, default_argument_parser, hooks, launch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from dataset_register import register_my_dataset

def setup_cfg(args):
    cfg = get_cfg()
    # Use Mask R-CNN config from model zoo (ResNet-50-FPN baseline)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (args.train_dataset_name,)
    cfg.DATASETS.TEST = (args.val_dataset_name,) if args.val_dataset_name else ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch  # images per batch across all GPUs
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # default 512 for FPN
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return cfg

def main(args):
    # register dataset
    register_my_dataset(args.train_dataset_name, args.train_json, args.train_image_dir)
    if args.val_json:
        register_my_dataset(args.val_dataset_name, args.val_json, args.val_image_dir)

    cfg = setup_cfg(args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--val_dataset_name", default="")
    parser.add_argument("--val_json", default="")
    parser.add_argument("--val_image_dir", default="")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--ims_per_batch", type=int, default=2)
    parser.add_argument("--base_lr", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=90000)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
