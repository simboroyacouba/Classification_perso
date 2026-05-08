"""
Inférence du classificateur personnalisé - Toitures cadastrales
Modes :
  1. Détection (défaut) : sliding window multi-échelle + NMS → bboxes + classes
  2. --bbox "x,y,w,h"  : classifie une région précise
  3. --dir              : détection sur tout un dossier

Les images > 2048px sont découpées en tuiles automatiquement.

Usage:
  python inference.py --image photo.jpg
  python inference.py --image photo.jpg --score-threshold 0.7
  python inference.py --dir dossier/images/
  python inference.py --image photo.jpg --bbox "10,20,200,150"
"""

import os
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from model import build_model

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

COLORS = {
    'panneau_solaire':       (255, 50,  50),
    'batiment_peint':        (50,  200, 50),
    'batiment_non_enduit':   (50,  50,  255),
    'batiment_enduit':       (255, 165, 0),
    'menuiserie_metallique': (128, 0,   128),
}

SCALES       = [0.15, 0.25, 0.40, 0.60]  # fraction de min(W, H)
STRIDE_RATIO = 0.5                         # stride = win_size * STRIDE_RATIO
MAX_DIM      = 2048                        # taille max avant découpage en tuiles
BATCH_SIZE   = 32


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_model(model_path, device):
    checkpoint       = torch.load(model_path, map_location=device, weights_only=False)
    num_classes      = checkpoint['num_classes']
    classes          = checkpoint['classes']          # classes objet uniquement
    background_label = checkpoint.get('background_label', None)
    image_size       = checkpoint.get('image_size', 299)
    model            = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, image_size, background_label


def get_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# CLASSIFICATION D'UN CROP
# =============================================================================

@torch.no_grad()
def classify_crop(model, crop: Image.Image, transform, device):
    tensor = transform(crop).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return probs


# =============================================================================
# SLIDING WINDOW + NMS  →  détection avec bboxes
# =============================================================================

def _tile_image(image: Image.Image, max_dim=MAX_DIM, overlap=256):
    """Génère des tuiles (tile, offset_x, offset_y) pour les grandes images."""
    img_w, img_h = image.size
    if img_w <= max_dim and img_h <= max_dim:
        yield image, 0, 0
        return
    step = max_dim - overlap
    for ty in range(0, img_h, step):
        for tx in range(0, img_w, step):
            x2 = min(tx + max_dim, img_w)
            y2 = min(ty + max_dim, img_h)
            yield image.crop((tx, ty, x2, y2)), tx, ty


@torch.no_grad()
def _flush_batch(model, crops, coords, offset_x, offset_y,
                 class_names, score_threshold, background_label,
                 out_boxes, out_scores, out_labels, out_probs, device):
    """Classifie un batch de crops et ajoute les résultats au-dessus du seuil.
    Les prédictions 'background' sont toujours rejetées."""
    batch  = torch.stack(crops).to(device)
    probs  = torch.softmax(model(batch), dim=1).cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(coords):
        pred_idx = int(np.argmax(probs[i]))
        # Rejeter la classe background
        if background_label is not None and pred_idx == background_label:
            continue
        score = float(probs[i][pred_idx])
        if score >= score_threshold:
            out_boxes.append([x1 + offset_x, y1 + offset_y,
                              x2 + offset_x, y2 + offset_y])
            out_scores.append(score)
            out_labels.append(pred_idx)
            out_probs.append(probs[i][:len(class_names)])  # exclure colonne bg


@torch.no_grad()
def detect_sliding_window(model, image: Image.Image, class_names, image_size,
                           device, score_threshold=0.6, nms_iou=0.3,
                           background_label=None):
    """
    Sliding window multi-échelle + NMS.
    Les images > MAX_DIM px sont découpées en tuiles avec chevauchement.
    Les prédictions 'background' sont rejetées si background_label est fourni.
    Retourne: liste de dicts {box, label, score, probs}
    """
    transform  = get_transform(image_size)
    all_boxes  = []
    all_scores = []
    all_labels = []
    all_probs  = []

    for tile, off_x, off_y in _tile_image(image):
        tw, th  = tile.size
        min_dim = min(tw, th)
        batch_crops  = []
        batch_coords = []

        for scale in SCALES:
            win_size = max(int(min_dim * scale), 32)
            stride   = max(int(win_size * STRIDE_RATIO), 16)
            for y in range(0, th - win_size + 1, stride):
                for x in range(0, tw - win_size + 1, stride):
                    crop = tile.crop((x, y, x + win_size, y + win_size))
                    batch_crops.append(transform(crop))
                    batch_coords.append((x, y, x + win_size, y + win_size))
                    if len(batch_crops) == BATCH_SIZE:
                        _flush_batch(model, batch_crops, batch_coords, off_x, off_y,
                                     class_names, score_threshold, background_label,
                                     all_boxes, all_scores, all_labels, all_probs, device)
                        batch_crops.clear()
                        batch_coords.clear()

        if batch_crops:
            _flush_batch(model, batch_crops, batch_coords, off_x, off_y,
                         class_names, score_threshold, background_label,
                         all_boxes, all_scores, all_labels, all_probs, device)

    if not all_boxes:
        return []

    # NMS par classe
    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    labels_t = np.array(all_labels)

    detections = []
    for cls_idx in range(len(class_names)):
        mask = labels_t == cls_idx
        if not mask.any():
            continue
        keep = ops.nms(boxes_t[mask], scores_t[mask], nms_iou)
        for i in np.where(mask)[0][keep.numpy()]:
            detections.append({
                'box':   [int(v) for v in all_boxes[i]],
                'label': class_names[all_labels[i]],
                'score': float(all_scores[i]),
                'probs': {class_names[j]: float(all_probs[i][j])
                          for j in range(len(class_names))},
            })

    detections.sort(key=lambda d: d['score'], reverse=True)
    return detections


# =============================================================================
# VISUALISATION
# =============================================================================

def draw_detections(image: Image.Image, detections, class_names):
    """Dessine les bboxes et labels sur l'image."""
    img_draw = image.copy()
    draw     = ImageDraw.Draw(img_draw)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = COLORS.get(det['label'], (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{det['label']} {det['score']*100:.0f}%"
        try:
            tw = draw.textlength(text, font=font)
        except Exception:
            tw = len(text) * 8
        draw.rectangle([x1, max(0, y1 - 18), x1 + int(tw) + 4, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 17)), text, fill=(255, 255, 255), font=font)

    return img_draw


def save_detection_figure(image_orig, detections, class_names, output_path):
    img_annotated = draw_detections(image_orig, detections, class_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_annotated)
    axes[0].set_title(f"Détections ({len(detections)} objets)")
    axes[0].axis('off')

    counts     = {n: 0 for n in class_names}
    for d in detections:
        counts[d['label']] += 1
    colors_bar = [(r/255, g/255, b/255)
                  for r, g, b in [COLORS.get(c, (150, 150, 150)) for c in class_names]]
    axes[1].bar(range(len(class_names)), [counts[c] for c in class_names], color=colors_bar)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=30, ha='right')
    axes[1].set_ylabel('Nombre de détections')
    axes[1].set_title('Détections par classe')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MODES D'INFÉRENCE
# =============================================================================

def infer_detect(model, image_path, class_names, image_size, device,
                 score_threshold=0.6, nms_iou=0.3, background_label=None,
                 output_dir=None):
    image = Image.open(image_path).convert("RGB")
    print(f"\n🔍 {os.path.basename(image_path)}  ({image.size[0]}×{image.size[1]}px)")

    detections = detect_sliding_window(model, image, class_names, image_size,
                                       device, score_threshold, nms_iou,
                                       background_label=background_label)

    if not detections:
        print(f"   Aucune détection (score_threshold={score_threshold})")
    else:
        print(f"   {len(detections)} détection(s):")
        for d in detections:
            b = d['box']
            print(f"   [{b[0]},{b[1]},{b[2]},{b[3]}]  {d['label']:<30} {d['score']*100:.1f}%")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_detection_figure(image, detections, class_names,
                              os.path.join(output_dir, f"detect_{stem}.png"))
        with open(os.path.join(output_dir, f"detect_{stem}.json"), 'w') as f:
            json.dump({'image': os.path.basename(image_path),
                       'detections': detections}, f, indent=2)
        print(f"   💾 {output_dir}/detect_{stem}.png")

    return detections


def infer_bbox(model, image_path, bbox, class_names, image_size, device, output_dir=None):
    image     = Image.open(image_path).convert("RGB")
    transform = get_transform(image_size)
    x, y, w, h = bbox
    img_w, img_h = image.size
    x1 = max(0, int(x));  y1 = max(0, int(y))
    x2 = min(img_w, int(x + w)); y2 = min(img_h, int(y + h))
    crop  = image.crop((x1, y1, x2, y2))
    probs = classify_crop(model, crop, transform, device)
    pred_idx   = int(np.argmax(probs))
    pred_name  = class_names[pred_idx]
    confidence = probs[pred_idx]

    print(f"\n🖼️  {os.path.basename(image_path)}  bbox=({x1},{y1},{x2},{y2})")
    print(f"   Prédit: {pred_name}  ({confidence*100:.1f}%)")
    for i in np.argsort(probs)[::-1]:
        print(f"   [{i}] {class_names[i]:<30} {probs[i]*100:5.1f}%")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(crop); axes[0].axis('off')
        axes[0].set_title(f"Prédit: {pred_name}\n({confidence*100:.1f}%)")
        colors_bar = [(r/255, g/255, b/255)
                      for r, g, b in [COLORS.get(c, (150,150,150)) for c in class_names]]
        axes[1].bar(range(len(class_names)), probs * 100, color=colors_bar)
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=30, ha='right')
        axes[1].set_ylim(0, 100); axes[1].set_ylabel('Probabilité (%)')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pred_{stem}.png"), dpi=150)
        plt.close()

    return pred_idx, pred_name, confidence, probs


def infer_directory(model, dir_path, class_names, image_size, device,
                    score_threshold=0.6, nms_iou=0.3, background_label=None,
                    output_dir=None):
    extensions  = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    if not image_files:
        print(f"Aucune image trouvée dans {dir_path}"); return []

    print(f"\n📂 Détection sur {len(image_files)} images...")
    all_results = []
    counts      = {n: 0 for n in class_names}

    for img_path in image_files:
        try:
            dets = infer_detect(model, img_path, class_names, image_size, device,
                                score_threshold, nms_iou, background_label, output_dir)
            for d in dets:
                counts[d['label']] += 1
            all_results.append({'image': os.path.basename(img_path), 'detections': dets})
        except Exception as e:
            print(f"   ⚠️ Erreur sur {img_path}: {e}")

    total = sum(counts.values())
    print(f"\n📊 Résumé — {total} détections sur {len(all_results)} images:")
    for name, count in counts.items():
        print(f"   {name:<30} {count:>4}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
            json.dump(all_results, f, indent=2)

    return all_results


# =============================================================================
# AUTO-DÉTECTION DU MODÈLE
# =============================================================================

def find_best_model():
    """Cherche le meilleur modele perso dans les dossiers de sortie du train.py."""
    env_path = os.getenv("MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    base_output = os.getenv("OUTPUT_DIR", "./output")
    search_dirs = [
        os.path.join(base_output, "runs", "classify"),          # unified
        os.path.join(base_output, "nadir", "runs", "classify"), # dual nadir
        os.path.join(base_output, "oblique", "runs", "classify"),# dual oblique
        "runs/classify/train",                                    # legacy
    ]
    for runs_dir in search_dirs:
        if not os.path.isdir(runs_dir):
            continue
        runs = sorted(
            [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
             if os.path.isdir(os.path.join(runs_dir, d))
             and d.startswith("perso_")],
            key=os.path.getmtime, reverse=True,
        )
        for run in runs:
            candidate = os.path.join(run, "best_model.pth")
            if os.path.exists(candidate):
                return candidate
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inférence EfficientNet-B3 Perso — Détection toitures")
    parser.add_argument('--model',           default=None,  help="Chemin vers best_model.pth (auto-détecté si omis)")
    parser.add_argument('--image',           default=None,  help="Image à traiter")
    parser.add_argument('--dir',             default=None,  help="Dossier d'images")
    parser.add_argument('--bbox',            default=None,  help="Classifier une région précise: 'x,y,w,h'")
    parser.add_argument('--score-threshold', default=0.6,   type=float, help="Seuil de confiance (défaut: 0.6)")
    parser.add_argument('--nms-iou',         default=0.3,   type=float, help="Seuil IoU pour NMS (défaut: 0.3)")
    parser.add_argument('--output',          default="./predictions", help="Dossier de sortie")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path:
        parser.error("Aucun modèle trouvé. Spécifier --model ou lancer train.py d'abord.")
    if not os.path.exists(model_path):
        parser.error(f"Modèle introuvable: {model_path}")
    if not args.model:
        print(f"   Modèle auto-détecté: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names, image_size, background_label = load_model(model_path, device)
    print(f"Classes: {class_names} | Image size: {image_size}px")
    if background_label is not None:
        print(f"Background label: {background_label} (rejeté automatiquement)")

    if args.image and args.bbox:
        bbox = list(map(float, args.bbox.split(',')))
        infer_bbox(model, args.image, bbox, class_names, image_size, device,
                   output_dir=args.output)
    elif args.image:
        infer_detect(model, args.image, class_names, image_size, device,
                     score_threshold=args.score_threshold,
                     nms_iou=args.nms_iou,
                     background_label=background_label,
                     output_dir=args.output)
    elif args.dir:
        infer_directory(model, args.dir, class_names, image_size, device,
                        score_threshold=args.score_threshold,
                        nms_iou=args.nms_iou,
                        background_label=background_label,
                        output_dir=args.output)
    else:
        # Auto-detection du dossier d'images
        auto_dir = None
        for candidate in (
            os.getenv("DETECTION_INFERENCE_IMAGES_DIR", ""),
            os.getenv("DETECTION_TEST_IMAGES_DIR", ""),
            "./test_images", "./images", "../test",
        ):
            if candidate and os.path.isdir(candidate):
                auto_dir = candidate
                break
        if auto_dir:
            print(f"   Dossier auto-detecte: {auto_dir}")
            infer_directory(model, auto_dir, class_names, image_size, device,
                            score_threshold=args.score_threshold,
                            nms_iou=args.nms_iou,
                            background_label=background_label,
                            output_dir=args.output)
        else:
            print("   Aucun dossier d'images trouve.")
            print("   Utilisez : python inference.py --image photo.jpg")
            print("              python inference.py --dir dossier/images/")
            print("   Ou definir DETECTION_INFERENCE_IMAGES_DIR dans .env")


if __name__ == "__main__":
    main()
