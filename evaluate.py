"""
Évaluation du classificateur personnalisé - Toitures cadastrales
Évaluation sur le TEST SET (10% du dataset)

Deux modes :
  1. Crops GT  : classifie les régions annotées → accuracy / F1 (tâche facile,
                 mesure la qualité du classifieur seul)
  2. Détection : sliding window sur les images → IoU matching → mAP@50
                 (comparable aux autres modèles SSD / YOLO / Faster R-CNN)

Usage :
  python evaluate.py                    # mode crops GT par défaut
  python evaluate.py --mode detect      # mode détection (mAP@50)
  python evaluate.py --mode both        # les deux
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
)
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from model import build_model
from train import CropDataset, get_val_transforms, format_time
from inference import detect_sliding_window, MAX_DIM
from pycocotools.coco import COCO
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CONFIG = {
    "model_path":        os.getenv("MODEL_PATH", None),
    "output_dir":        os.getenv("EVALUATION_DIR", "./evaluation"),
    "classes_file":      os.getenv("CLASSES_FILE", "classes.yaml"),
    "batch_size":        int(os.getenv("BATCH_SIZE", "16")),
    "score_threshold":   float(os.getenv("SCORE_THRESHOLD", "0.5")),
    "nms_iou":           float(os.getenv("NMS_IOU", "0.3")),
    "iou_threshold_map": float(os.getenv("IOU_THRESHOLD_MAP", "0.5")),
}


# =============================================================================
# AUTO-DÉTECTION DU MODÈLE
# =============================================================================

def find_best_model(runs_dir="runs/classify/train"):
    if not os.path.isdir(runs_dir):
        return None
    runs = sorted(
        [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
         if os.path.isdir(os.path.join(runs_dir, d))],
        key=os.path.getmtime, reverse=True
    )
    for run in runs:
        candidate = os.path.join(run, "best_model.pth")
        if os.path.exists(candidate):
            return candidate
    return None


# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================

def load_model(model_path, device):
    print(f"🧠 Chargement du modèle: {model_path}")
    checkpoint       = torch.load(model_path, map_location=device, weights_only=False)
    num_classes      = checkpoint['num_classes']
    classes          = checkpoint['classes']           # classes objet uniquement
    cat_mapping      = {int(k): v for k, v in checkpoint['cat_mapping'].items()}
    background_label = checkpoint.get('background_label', None)
    image_size       = checkpoint.get('image_size', 299)

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"   ✅ Époque {checkpoint['epoch']} | val_acc={checkpoint['val_acc']:.4f}")
    print(f"   Classes:     {classes}")
    print(f"   Cat mapping: {cat_mapping}")
    if background_label is not None:
        print(f"   Background:  label {background_label} (rejeté en détection)")
    return model, classes, cat_mapping, image_size, background_label


# =============================================================================
# CHARGEMENT DU TEST SET
# =============================================================================

def load_test_info(model_path):
    """Cherche test_info.json dans le dossier du run."""
    for candidate_dir in [os.path.dirname(model_path),
                          os.path.dirname(os.path.dirname(model_path))]:
        path = os.path.join(candidate_dir, "test_info.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("test_info.json introuvable. Relancer train.py d'abord.")


def resolve_paths(test_info):
    """
    Retourne (images_dir, annotations_file) depuis test_info.json.
    Vérifie l'existence et propose le fallback .env si le chemin est mort.
    """
    images_dir       = test_info['images_dir']
    annotations_file = test_info['annotations_file']

    if not os.path.isdir(images_dir):
        fallback = os.getenv("DETECTION_DATASET_IMAGES_DIR", "")
        if fallback and os.path.isdir(fallback):
            print(f"   ⚠️  images_dir du checkpoint introuvable, fallback .env: {fallback}")
            images_dir = fallback
        else:
            raise FileNotFoundError(
                f"Dossier images introuvable: {images_dir}\n"
                f"Définir DETECTION_DATASET_IMAGES_DIR dans .env"
            )

    if not os.path.exists(annotations_file):
        fallback = os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "")
        if fallback and os.path.exists(fallback):
            print(f"   ⚠️  annotations_file du checkpoint introuvable, fallback .env: {fallback}")
            annotations_file = fallback
        else:
            raise FileNotFoundError(
                f"Fichier annotations introuvable: {annotations_file}\n"
                f"Définir DETECTION_DATASET_ANNOTATIONS_FILE dans .env"
            )

    return images_dir, annotations_file


# =============================================================================
# MODE 1 : ÉVALUATION SUR CROPS GT
# =============================================================================

@torch.no_grad()
def evaluate_crops(model, dataloader, class_names, device):
    all_preds  = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def run_crop_evaluation(model, images_dir, annotations_file, test_image_ids,
                        cat_mapping, class_names, image_size, min_crop_size,
                        device, output_dir):
    print("\n" + "=" * 70)
    print("   MODE 1 — Crops GT (précision classifieur)")
    print("=" * 70)

    test_dataset = CropDataset(
        images_dir, annotations_file,
        test_image_ids, cat_mapping, min_crop_size,
        transform=get_val_transforms()
    )

    # Diagnostic : répartition des crops par classe
    label_counts = defaultdict(int)
    missing_imgs = 0
    for s in test_dataset.samples:
        label_counts[s['label']] += 1
    for img_id in test_image_ids:
        coco = COCO(annotations_file)
        info = coco.imgs.get(img_id, {})
        path = os.path.join(images_dir, info.get('file_name', ''))
        if not os.path.exists(path):
            missing_imgs += 1

    print(f"\n   Crops test chargés: {len(test_dataset)}")
    if missing_imgs:
        print(f"   ⚠️  {missing_imgs} images introuvables sur disque (crops ignorés)")
    print(f"   Distribution par classe:")
    for cls_idx, cls_name in enumerate(class_names):
        count = label_counts.get(cls_idx, 0)
        ok    = "⚠️ VIDE" if count == 0 else ""
        print(f"      [{cls_idx}] {cls_name:<30} {count:>5} crops {ok}")

    if len(test_dataset) == 0:
        print("\n   ❌ Aucun crop trouvé — vérifier les chemins dans .env")
        return None

    loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=0)

    t0 = time.time()
    labels, preds = evaluate_crops(model, loader, class_names, device)
    elapsed = time.time() - t0

    acc       = accuracy_score(labels, preds)
    f1_macro  = f1_score(labels, preds, average='macro',    zero_division=0)
    f1_weight = f1_score(labels, preds, average='weighted', zero_division=0)
    prec      = precision_score(labels, preds, average='macro', zero_division=0)
    rec       = recall_score(labels, preds, average='macro',    zero_division=0)
    cm        = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report    = classification_report(labels, preds, target_names=class_names,
                                      labels=list(range(len(class_names))), zero_division=0)

    print(f"\n   Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1 macro:        {f1_macro:.4f}")
    print(f"   F1 weighted:     {f1_weight:.4f}")
    print(f"   Précision macro: {prec:.4f}")
    print(f"   Rappel macro:    {rec:.4f}")
    print(f"   Temps:           {format_time(elapsed)} ({len(test_dataset)} crops)")
    print(f"\n{report}")

    # Matrice de confusion
    _save_confusion_matrix(cm, class_names,
                           os.path.join(output_dir, "confusion_matrix_crops.png"),
                           title="Matrice de Confusion — Crops GT")

    return {
        'mode': 'crops_gt',
        'num_crops': len(test_dataset),
        'accuracy': float(acc), 'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weight), 'precision_macro': float(prec),
        'recall_macro': float(rec), 'confusion_matrix': cm.tolist(),
        'classification_report': report, 'inference_time_seconds': float(elapsed),
    }


# =============================================================================
# MODE 2 : DÉTECTION — SLIDING WINDOW → mAP@50
# =============================================================================

def _iou(box_a, box_b):
    """box = [x1, y1, x2, y2]"""
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    denom = a + b - inter
    return inter / denom if denom > 0 else 0.0


def _compute_ap(precisions, recalls):
    """Calcule l'AP avec interpolation sur 11 points."""
    return sum(
        np.max(precisions[recalls >= t]) if (recalls >= t).any() else 0.0
        for t in np.arange(0, 1.1, 0.1)
    ) / 11


def compute_map(all_preds, all_gts, class_names, iou_threshold=0.5):
    """
    all_preds : liste de {boxes:[[x1,y1,x2,y2],...], labels:[str,...], scores:[float,...]}
    all_gts   : liste de {boxes:[[x1,y1,x2,y2],...], labels:[str,...]}
    """
    aps = {}
    for cls_name in class_names:
        tps, fps, scores_list = [], [], []
        n_gt = sum(gt['labels'].count(cls_name) for gt in all_gts)
        if n_gt == 0:
            continue
        for pred, gt in zip(all_preds, all_gts):
            p_mask   = [l == cls_name for l in pred['labels']]
            g_boxes  = [b for b, l in zip(gt['boxes'], gt['labels']) if l == cls_name]
            p_boxes  = [b for b, m in zip(pred['boxes'], p_mask) if m]
            p_scores = [s for s, m in zip(pred['scores'], p_mask) if m]
            matched  = set()
            for idx in np.argsort(-np.array(p_scores)) if p_scores else []:
                scores_list.append(p_scores[idx])
                if not g_boxes:
                    tps.append(0); fps.append(1); continue
                ious     = [_iou(p_boxes[idx], g) for g in g_boxes]
                best_j   = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j); tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)

        if not scores_list:
            aps[cls_name] = 0.0; continue

        order  = np.argsort(-np.array(scores_list))
        tp_cum = np.cumsum(np.array(tps)[order])
        fp_cum = np.cumsum(np.array(fps)[order])
        prec   = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec    = tp_cum / (n_gt + 1e-10)
        aps[cls_name] = _compute_ap(prec, rec)

    return aps


def compute_detection_prf(all_preds, all_gts, class_names, iou_threshold=0.5):
    """
    Précision, Rappel et F1 par classe à seuil de score fixe (après NMS).
    - TP : prédiction matchée à un GT de même classe avec IoU >= iou_threshold
    - FP : prédiction non matchée
    - FN : GT non matchée par aucune prédiction
    Retourne (per_class_dict, macro_precision, macro_recall, macro_f1)
    """
    per_class = {}
    for cls_name in class_names:
        tp = fp = fn = 0
        for pred, gt in zip(all_preds, all_gts):
            p_boxes = [b for b, l in zip(pred['boxes'], pred['labels']) if l == cls_name]
            g_boxes = [b for b, l in zip(gt['boxes'],  gt['labels'])  if l == cls_name]
            matched_gt   = set()
            matched_pred = set()
            for pi, pb in enumerate(p_boxes):
                if not g_boxes:
                    fp += 1; continue
                ious   = [_iou(pb, gb) for gb in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched_gt:
                    matched_gt.add(best_j)
                    matched_pred.add(pi)
                    tp += 1
                else:
                    fp += 1
            fn += len(g_boxes) - len(matched_gt)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls_name] = {'precision': prec, 'recall': rec, 'f1': f1,
                               'tp': tp, 'fp': fp, 'fn': fn}

    valid    = [v for v in per_class.values()]
    macro_p  = float(np.mean([v['precision'] for v in valid])) if valid else 0.0
    macro_r  = float(np.mean([v['recall']    for v in valid])) if valid else 0.0
    macro_f1 = float(np.mean([v['f1']        for v in valid])) if valid else 0.0
    return per_class, macro_p, macro_r, macro_f1


def run_detection_evaluation(model, coco, images_dir, test_image_ids,
                              cat_mapping, class_names, image_size, device,
                              score_threshold, nms_iou, iou_threshold,
                              background_label, output_dir):
    print("\n" + "=" * 70)
    print("   MODE 2 — Détection sliding window (mAP@50)")
    print("=" * 70)

    # Mapping inverse : cat_id → nom de classe
    catid_to_name = {cat_id: class_names[idx] for cat_id, idx in cat_mapping.items()}

    all_preds = []
    all_gts   = []
    t0        = time.time()

    for i, img_id in enumerate(test_image_ids):
        img_info = coco.imgs.get(img_id, {})
        img_path = os.path.join(images_dir, img_info.get('file_name', ''))
        if not os.path.exists(img_path):
            continue

        # Ground truth
        gt_boxes  = []
        gt_labels = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            if ann.get('iscrowd', 0):
                continue
            name = catid_to_name.get(ann['category_id'])
            if name is None:
                continue
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(name)

        if not gt_boxes:
            continue

        # Détection
        image = Image.open(img_path).convert("RGB")
        dets  = detect_sliding_window(model, image, class_names, image_size,
                                      device, score_threshold, nms_iou,
                                      background_label=background_label)

        all_gts.append({'boxes': gt_boxes, 'labels': gt_labels})
        all_preds.append({
            'boxes':  [d['box'] for d in dets],
            'labels': [d['label'] for d in dets],
            'scores': [d['score'] for d in dets],
        })

        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(test_image_ids)} images traitées...")

    elapsed = time.time() - t0

    if not all_gts:
        print("   ❌ Aucune image de test trouvée sur disque.")
        return None

    aps                    = compute_map(all_preds, all_gts, class_names, iou_threshold)
    map50                  = float(np.mean(list(aps.values()))) if aps else 0.0
    per_class_prf, macro_p, macro_r, macro_f1 = compute_detection_prf(
        all_preds, all_gts, class_names, iou_threshold
    )

    iou_pct = int(iou_threshold * 100)
    print(f"\n   mAP@{iou_pct}:          {map50:.4f} ({map50*100:.2f}%)")
    print(f"   Précision macro: {macro_p:.4f}")
    print(f"   Rappel macro:    {macro_r:.4f}")
    print(f"   F1 macro:        {macro_f1:.4f}")
    print(f"\n   {'Classe':<30} {'AP':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"   {'-'*58}")
    for name in class_names:
        ap  = aps.get(name, 0.0)
        prf = per_class_prf.get(name, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
        print(f"   {name:<30} {ap:>7.4f} {prf['precision']:>7.4f} {prf['recall']:>7.4f} {prf['f1']:>7.4f}")
    print(f"\n   Temps: {format_time(elapsed)} ({len(all_gts)} images)")

    return {
        'mode':            'detection',
        'num_test_images': len(all_gts),
        'map50':           map50,
        'precision_macro': float(macro_p),
        'recall_macro':    float(macro_r),
        'f1_macro':        float(macro_f1),
        'ap_per_class':    aps,
        'per_class_metrics': per_class_prf,
        'iou_threshold':   iou_threshold,
        'score_threshold': score_threshold,
        'inference_time_seconds': float(elapsed),
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def _save_confusion_matrix(cm, class_names, path, title="Matrice de Confusion"):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def evaluate(mode='both', score_threshold=None, nms_iou=None):
    model_path = CONFIG["model_path"] or find_best_model()
    if not model_path:
        raise FileNotFoundError(
            "Aucun modèle trouvé. Définir MODEL_PATH dans .env ou lancer train.py."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    if not CONFIG["model_path"]:
        print(f"   Modèle auto-détecté: {model_path}")

    score_threshold = score_threshold or CONFIG["score_threshold"]
    nms_iou         = nms_iou         or CONFIG["nms_iou"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names, cat_mapping, image_size, background_label = load_model(model_path, device)
    num_classes = len(class_names)

    test_info        = load_test_info(model_path)
    images_dir, annotations_file = resolve_paths(test_info)
    test_image_ids   = test_info['test_image_ids']
    min_crop_size    = test_info.get('min_crop_size', 10)

    print(f"\n📂 Test set: {len(test_image_ids)} images | {images_dir}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG["output_dir"], f"eval_{mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    coco    = COCO(annotations_file)

    if mode in ('crops', 'both'):
        r = run_crop_evaluation(
            model, images_dir, annotations_file, test_image_ids,
            cat_mapping, class_names, image_size, min_crop_size, device, output_dir
        )
        if r:
            results['crops'] = r

    if mode in ('detect', 'both'):
        r = run_detection_evaluation(
            model, coco, images_dir, test_image_ids,
            cat_mapping, class_names, image_size, device,
            score_threshold, nms_iou, CONFIG["iou_threshold_map"],
            background_label, output_dir
        )
        if r:
            results['detection'] = r

    # Sauvegarde
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   📁 Résultats: {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation EfficientNet-B3 Perso")
    parser.add_argument('--mode',            default='both',
                        choices=['crops', 'detect', 'both'],
                        help="Mode d'évaluation (défaut: both)")
    parser.add_argument('--score-threshold', default=None, type=float,
                        help="Seuil confiance pour détection (défaut: .env SCORE_THRESHOLD)")
    parser.add_argument('--nms-iou',         default=None, type=float,
                        help="Seuil IoU NMS (défaut: .env NMS_IOU)")
    args = parser.parse_args()
    evaluate(mode=args.mode,
             score_threshold=args.score_threshold,
             nms_iou=args.nms_iou)
