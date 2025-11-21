# infer_pspnet.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import PSPNet
import argparse
import os

def inference(image_path, model_path, num_classes, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSPNet(n_classes=num_classes, backbone='resnet50', pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    transform = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    # simple color mapping (one channel)
    Image.fromarray(pred).save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--out", default="psp_pred.png")
    args = parser.parse_args()
    inference(args.image, args.model, args.num_classes, args.out)
