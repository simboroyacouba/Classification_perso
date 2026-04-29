"""
Inférence du classificateur personnalisé - Toitures cadastrales
Modes :
  1. Image complète  → classifie l'image entière
  2. Dossier         → classifie toutes les images du dossier
  3. Avec bboxes     → classifie des régions extraites (JSON bbox)

Usage:
  python inference.py --model runs/.../best_model.pth --image chemin/image.jpg
  python inference.py --model runs/.../best_model.pth --dir chemin/dossier/
  python inference.py --model runs/.../best_model.pth --image img.jpg --bbox "10,20,200,150"
"""

import os
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from model import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

COLORS = {
    'panneau_solaire':    (255, 0,   0),
    'batiment_peint':     (0,   200, 0),
    'batiment_non_enduit':(0,   0,   255),
    'batiment_enduit':    (255, 165, 0),
}


def load_model(model_path, device):
    checkpoint  = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    classes     = checkpoint['classes']
    image_size  = checkpoint.get('image_size', 299)

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, image_size


def get_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def classify_crop(model, crop: Image.Image, transform, device):
    tensor = transform(crop).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def format_result(class_names, probs, top_k=3):
    order = np.argsort(probs)[::-1]
    lines = []
    for i in range(min(top_k, len(class_names))):
        idx = order[i]
        lines.append(f"  [{idx}] {class_names[idx]:<30} {probs[idx]*100:5.1f}%")
    return "\n".join(lines)


def infer_image(model, image_path, class_names, image_size, device,
                bbox=None, output_dir=None):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(image_size)

    if bbox:
        x, y, w, h = bbox
        img_w, img_h = image.size
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))
        crop = image.crop((x1, y1, x2, y2))
    else:
        crop = image

    probs     = classify_crop(model, crop, transform, device)
    pred_idx  = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = probs[pred_idx]

    print(f"\n🖼️  {os.path.basename(image_path)}")
    if bbox:
        print(f"   Bbox: {bbox}")
    print(f"   Prédit: {pred_name}  ({confidence*100:.1f}%)")
    print(f"   Top-3:")
    print(format_result(class_names, probs))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(crop)
        axes[0].set_title(f"Crop / Image\nPrédit: {pred_name} ({confidence*100:.1f}%)")
        axes[0].axis('off')

        colors_bar = [COLORS.get(c, (100, 100, 100)) for c in class_names]
        colors_bar = [(r/255, g/255, b/255) for r, g, b in colors_bar]
        bars = axes[1].bar(range(len(class_names)), probs * 100, color=colors_bar)
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=30, ha='right')
        axes[1].set_ylabel('Probabilité (%)')
        axes[1].set_ylim(0, 100)
        axes[1].set_title('Distribution des probabilités')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out_path = os.path.join(output_dir,
                                f"pred_{os.path.splitext(os.path.basename(image_path))[0]}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"   💾 Visualisation: {out_path}")

    return pred_idx, pred_name, confidence, probs


def infer_directory(model, dir_path, class_names, image_size, device, output_dir=None):
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    if not image_files:
        print(f"Aucune image trouvée dans {dir_path}")
        return []

    print(f"\n📂 Inférence sur {len(image_files)} images...")
    results = []
    counts  = {name: 0 for name in class_names}

    for img_path in image_files:
        try:
            pred_idx, pred_name, conf, probs = infer_image(
                model, img_path, class_names, image_size, device, output_dir=output_dir
            )
            counts[pred_name] += 1
            results.append({
                'image':      os.path.basename(img_path),
                'predicted':  pred_name,
                'confidence': float(conf),
                'probs':      {class_names[i]: float(probs[i]) for i in range(len(class_names))},
            })
        except Exception as e:
            print(f"   ⚠️ Erreur sur {img_path}: {e}")

    print(f"\n📊 Résumé ({len(results)} images):")
    for name, count in counts.items():
        pct = count / max(len(results), 1) * 100
        print(f"   {name:<30} {count:>4} ({pct:.1f}%)")

    if output_dir:
        with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Inférence EfficientNet-B3 Perso")
    parser.add_argument('--model', required=True, help="Chemin vers best_model.pth")
    parser.add_argument('--image', default=None, help="Image à classifier")
    parser.add_argument('--dir',   default=None, help="Dossier d'images")
    parser.add_argument('--bbox',  default=None,
                        help="Bbox à cropper: 'x,y,w,h' en pixels")
    parser.add_argument('--output', default="./predictions",
                        help="Dossier de sortie pour les visualisations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names, image_size = load_model(args.model, device)
    print(f"Classes: {class_names} | Image size: {image_size}px")

    bbox = None
    if args.bbox:
        bbox = list(map(float, args.bbox.split(',')))

    if args.image:
        infer_image(model, args.image, class_names, image_size, device,
                    bbox=bbox, output_dir=args.output)
    elif args.dir:
        infer_directory(model, args.dir, class_names, image_size, device,
                        output_dir=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
