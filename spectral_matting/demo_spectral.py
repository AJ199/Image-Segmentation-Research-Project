# demo_spectral.py
from spectral_matting import segment_image
import matplotlib.pyplot as plt
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="spectral_seg.png")
    parser.add_argument("--n_segments", type=int, default=400)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    seg = segment_image(args.image, n_segments=args.n_segments, compactness=args.compactness, threshold=args.threshold)
    plt.figure(figsize=(10,10))
    plt.imshow(seg)
    plt.axis('off')
    plt.savefig(args.out, bbox_inches='tight')
    print("Saved:", args.out)
