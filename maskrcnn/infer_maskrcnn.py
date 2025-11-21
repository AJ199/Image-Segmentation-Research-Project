# infer_maskrcnn.py
import cv2
import os
import argparse
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

def get_predictor(weights, num_classes, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.freeze()
    return DefaultPredictor(cfg)

def visualize_and_save(img_path, predictor, output_path, metadata=None):
    img = cv2.imread(img_path)[:, :, ::-1]
    outputs = predictor(img)
    v = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = v.get_image()[:, :, ::-1]
    cv2.imwrite(output_path, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--out", default="out.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predictor = get_predictor(args.weights, args.num_classes, args.threshold)
    metadata = None
    visualize_and_save(args.image, predictor, args.out, metadata)
    print("Saved:", args.out)
