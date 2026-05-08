"""
Entraînement du classificateur personnalisé - Toitures cadastrales
Dataset : Images aériennes annotées avec CVAT (format COCO)
          → chaque bbox est cropée et classifiée (pas de détection)
Classes : Chargées depuis classes.yaml
Config  : Chargée depuis .env

Architecture : EfficientNet-B3 + ResidualAttentionBlock + CustomPoolingConcatenate
Stratégie    : Phase 1 backbone gelé (head only) → Phase 2 fine-tuning complet

Modes :
  simple   : entraînement standard sur toutes les classes
  attention: entraînement avec bloc CBAM dans le backbone
  optimize : recherche Optuna des hyperparamètres, puis entraînement final
  dual     : entraînement séquentiel nadir (panneau_solaire) + oblique (batiments)

Usage :
  python train.py --mode simple
  python train.py --mode attention --cbam-reduction 16
  python train.py --mode optimize --n-trials 20
  python train.py --mode dual --aug panneau_solaire:3
"""

import os
import copy
import json
import yaml
import shutil
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

from model import build_model

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONSTANTES
# =============================================================================

MODE_CLASSES = {
    "nadir":   ["panneau_solaire"],
    "oblique": [
        "batiment_peint", "batiment_non_enduit",
        "batiment_enduit", "menuiserie_metallique",
    ],
}

OPTUNA_CONFIG = {
    "n_trials":           20,
    "n_epochs_per_trial": 5,
    "study_name":         "perso_cadastral",
    "output_dir":         "./optuna_output",
}

IMAGE_SIZE = 299  # Taille fixe EfficientNet-B3


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml", mode_classes=None):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    all_classes = [c for c in data.get('classes', []) if c != '__background__']
    if mode_classes is not None:
        classes = [c for c in all_classes if c in mode_classes]
        if not classes:
            classes = list(mode_classes)
    else:
        classes = all_classes
    print(f"Classes depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# CONFIGURATION
# =============================================================================

def _base_config():
    return {
        "images_dir":       os.getenv("DETECTION_DATASET_IMAGES_DIR",       "../dataset1/images/default"),
        "annotations_file": os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
        "output_dir":       os.getenv("OUTPUT_DIR",   "./output"),
        "classes_file":     os.getenv("CLASSES_FILE", "classes.yaml"),
        "classes":          None,
        "num_epochs":       int(os.getenv("NUM_EPOCHS",    "60")),
        "batch_size":       int(os.getenv("BATCH_SIZE",    "16")),
        "learning_rate":    float(os.getenv("LEARNING_RATE", "1e-3")),
        "weight_decay":     float(os.getenv("WEIGHT_DECAY",  "1e-4")),
        "train_split":      float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":        float(os.getenv("VAL_SPLIT",   "0.20")),
        "test_split":       float(os.getenv("TEST_SPLIT",  "0.10")),
        "save_every":       int(os.getenv("SAVE_EVERY",    "5")),
        "pretrained":       os.getenv("PRETRAINED", "true").lower() == "true",
        "min_crop_size":    int(os.getenv("MIN_CROP_SIZE", "10")),
        "freeze_epochs":    int(os.getenv("FREEZE_EPOCHS", "10")),
        "lr_backbone":      float(os.getenv("LR_BACKBONE",  "1e-5")),
        "bg_crops_per_img": int(os.getenv("BG_CROPS_PER_IMG", "3")),
        "bg_iou_threshold": float(os.getenv("BG_IOU_THRESHOLD", "0.1")),
        "aug_brightness":   float(os.getenv("AUG_BRIGHTNESS", "0.113")),
        "aug_saturation":   float(os.getenv("AUG_SATURATION", "0.162")),
        "aug_hue":          float(os.getenv("AUG_HUE",        "0.05")),
    }


def _build_sub_config(base_config, mode_label):
    """Construit un sous-config nadir ou oblique depuis le config de base."""
    ann_file = base_config["annotations_file"]
    ann_dir  = os.path.dirname(os.path.abspath(ann_file))
    cfg = copy.deepcopy(base_config)
    cfg["mode"] = mode_label
    if mode_label == "nadir":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_nadir.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "nadir")
    elif mode_label == "oblique":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_oblique.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "oblique")
    return cfg


# =============================================================================
# PARSE AUG COEFFICIENTS
# =============================================================================

def parse_aug_coeffs(aug_args, classes):
    """
    Analyse --aug CLASS:COEFF avec correspondance partielle insensible à la casse.
    Retourne un dict {class_name: coeff} (int >= 1) pour les classes connues.
    Utilisé comme multiplicateur d'augmentation par classe.
    """
    coeffs = {}
    for entry in (aug_args or []):
        if ':' not in entry:
            print(f"   [WARN] --aug '{entry}' ignoré (format attendu CLASS:COEFF)")
            continue
        raw_name, raw_coeff = entry.rsplit(':', 1)
        try:
            coeff = int(raw_coeff)
        except ValueError:
            print(f"   [WARN] --aug '{entry}' ignoré (coefficient non entier)")
            continue
        if coeff < 1:
            print(f"   [WARN] --aug '{entry}' ignoré (coefficient < 1)")
            continue
        raw_lower = raw_name.lower()
        matched = [c for c in classes if raw_lower in c.lower()]
        if not matched:
            print(f"   [WARN] --aug '{raw_name}' ne correspond à aucune classe connue")
            continue
        if len(matched) > 1:
            print(f"   [WARN] --aug '{raw_name}' ambigu ({matched}), ignoré")
            continue
        coeffs[matched[0]] = coeff
    return coeffs


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def stratified_split(coco, train_split, val_split, test_split, seed=42):
    np.random.seed(seed)
    all_image_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_image_ids)
    n_total = len(all_image_ids)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val
    if n_test < 1 and n_total > 2:
        n_test  = max(1, int(n_total * 0.10))
        n_train = n_total - n_val - n_test

    print(f"\n   Split des IMAGES (total: {n_total}):")
    print(f"      Train: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"      Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"      Test:  {n_test}  ({n_test/n_total*100:.1f}%)")

    train_ids = all_image_ids[:n_train]
    val_ids   = all_image_ids[n_train:n_train + n_val]
    test_ids  = all_image_ids[n_train + n_val:]

    stats = {split: {cat_id: 0 for cat_id in coco.getCatIds()}
             for split in ('train', 'val', 'test')}
    for split, ids in (('train', train_ids), ('val', val_ids), ('test', test_ids)):
        for img_id in ids:
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                if ann['category_id'] in stats[split]:
                    stats[split][ann['category_id']] += 1

    return train_ids, val_ids, test_ids, stats


def print_split_stats(coco, stats, class_names):
    print("\n   Distribution des crops par classe (split 70/20/10):")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = " !" if val == 0 or test == 0 else ""
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8}{ok}")
    print(f"   {'-'*70}")


# =============================================================================
# DATASET — CROPS DE BBOXES COCO
# =============================================================================

def get_train_transforms(brightness=0.113, saturation=0.162, hue=0.05):
    return T.Compose([
        T.ColorJitter(brightness=brightness, saturation=saturation, hue=hue),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_class_transforms(class_names, class_weights,
                          brightness=0.113, saturation=0.162, hue=0.05):
    """Retourne {label_idx: transform} avec multiplicateur par classe."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transforms = {}
    for idx, name in enumerate(class_names):
        w = float(class_weights.get(name, 1.0))
        if w > 0:
            transforms[idx] = T.Compose([
                T.ColorJitter(brightness=brightness * w,
                              saturation=saturation * w,
                              hue=min(hue * w, 0.5)),
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            transforms[idx] = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
    return transforms


def _iou_xywh(box_a, box_b):
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def _mine_background_crops(img_w, img_h, gt_bboxes, n=3,
                            iou_threshold=0.1, min_size=32, seed=None):
    rng   = np.random.RandomState(seed)
    crops = []
    for _ in range(n * 20):
        if len(crops) >= n:
            break
        scale = rng.uniform(0.05, 0.4)
        w     = max(min_size, int(min(img_w, img_h) * scale))
        h     = max(min_size, int(min(img_w, img_h) * scale * rng.uniform(0.7, 1.4)))
        w     = min(w, img_w);  h = min(h, img_h)
        x     = rng.randint(0, max(1, img_w - w))
        y     = rng.randint(0, max(1, img_h - h))
        crop_box = (x, y, w, h)
        max_iou  = max((_iou_xywh(crop_box, gt) for gt in gt_bboxes), default=0.0)
        if max_iou < iou_threshold:
            crops.append(crop_box)
    return crops


class CropDataset(Dataset):
    """
    Extrait et classifie les crops depuis un dataset COCO.
    - Crops positifs  : regions annotées (classes 0..N-1)
    - Crops background: regions aléatoires sans objet (classe N = background)
    bg_crops_per_img=0 désactive le background mining.
    """

    def __init__(self, images_dir, annotations_file, image_ids,
                 cat_mapping, min_crop_size=10, transform=None,
                 bg_crops_per_img=0, bg_iou_threshold=0.1,
                 class_transforms=None):
        self.images_dir       = images_dir
        self.coco             = COCO(annotations_file)
        self.cat_mapping      = cat_mapping
        self.min_crop_size    = min_crop_size
        self.transform        = transform
        self.class_transforms = class_transforms
        self.bg_crops_per_img = bg_crops_per_img
        self.bg_iou_threshold = bg_iou_threshold

        max_idx = max(cat_mapping.values()) if cat_mapping else -1
        if max_idx < 0:
            raise ValueError("cat_mapping vide")
        self.num_object_classes = max_idx + 1
        self.background_label   = self.num_object_classes
        self.samples            = self._build_samples(image_ids)

    def _build_samples(self, image_ids):
        samples = []
        for img_id in image_ids:
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.images_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                continue

            gt_bboxes = []
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
                if ann.get('iscrowd', 0):
                    continue
                class_idx = self.cat_mapping.get(ann['category_id'])
                if class_idx is None:
                    continue
                x, y, w, h = ann['bbox']
                if w < self.min_crop_size or h < self.min_crop_size:
                    continue
                gt_bboxes.append((x, y, w, h))
                samples.append({
                    'img_path': img_path,
                    'bbox':     (x, y, w, h),
                    'label':    class_idx,
                    'image_id': img_id,
                })

            if self.bg_crops_per_img > 0 and gt_bboxes:
                img_w    = img_info['width']
                img_h    = img_info['height']
                bg_crops = _mine_background_crops(
                    img_w, img_h, gt_bboxes,
                    n=self.bg_crops_per_img,
                    iou_threshold=self.bg_iou_threshold,
                    seed=img_id,
                )
                for bbox in bg_crops:
                    samples.append({
                        'img_path': img_path,
                        'bbox':     bbox,
                        'label':    self.background_label,
                        'image_id': img_id,
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s        = self.samples[idx]
        image    = Image.open(s['img_path']).convert("RGB")
        x, y, w, h = s['bbox']
        img_w, img_h = image.size
        x1 = max(0, int(x)); y1 = max(0, int(y))
        x2 = min(img_w, int(x + w)); y2 = min(img_h, int(y + h))
        crop  = image.crop((x1, y1, x2, y2))
        label = s['label']
        if self.class_transforms is not None:
            tf = self.class_transforms.get(label, self.transform)
        else:
            tf = self.transform
        if tf:
            crop = tf(crop)
        return crop, label


# =============================================================================
# MÉTRIQUES
# =============================================================================

def compute_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n        = max(len(dataloader), 1)
    avg_loss = total_loss / n
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))

    per_class_acc = {}
    for cls_idx in range(num_classes):
        mask = np.array(all_labels) == cls_idx
        if mask.sum() > 0:
            per_class_acc[cls_idx] = float(np.array(all_preds)[mask] == cls_idx) if mask.sum() == 1 \
                else float((np.array(all_preds)[mask] == cls_idx).mean())

    return avg_loss, float(acc), per_class_acc, all_preds, all_labels


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    n = max(len(dataloader), 1)
    return total_loss / n, correct / max(total, 1)


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def _train_single(config, aug_coeffs, mode_label, training_mode,
                  cbam_reduction=16, cbam_kernel_size=7):
    """
    Lance un entraînement complet du classificateur perso.
    training_mode: 'simple' | 'attention'
    """
    class_names = config["classes"]
    bg_per_img  = config["bg_crops_per_img"]
    num_classes = len(class_names) + (1 if bg_per_img > 0 else 0)
    all_cls     = class_names + (['background'] if bg_per_img > 0 else [])

    attention = 'cbam' if training_mode == 'attention' else None

    print("=" * 70)
    print(f"   EfficientNet-B3 Perso — Mode : {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"\n   Images:      {config['images_dir']}")
    print(f"   Annotations: {config['annotations_file']}")
    print(f"   Classes:     {num_classes} ({', '.join(all_cls)})")
    print(f"   Background:  {bg_per_img} crops/image (IoU < {config['bg_iou_threshold']})")
    print(f"   Epochs:      {config['num_epochs']} | Batch: {config['batch_size']} | LR: {config['learning_rate']}")
    print(f"   Phase 1:     {config['freeze_epochs']} epochs backbone gelé")
    print(f"   Attention:   {attention or 'none'}")
    if cbam_reduction and attention:
        print(f"   CBAM:        reduction={cbam_reduction}, kernel={cbam_kernel_size}")
    if aug_coeffs:
        print(f"   Aug coeffs:  {aug_coeffs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join(config.get("output_dir", "output"), "runs", "classify",
                               f"perso_{mode_label}_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    coco = COCO(config["annotations_file"])

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    cat_mapping = {}
    for cat_id in coco.getCatIds():
        cat_name = coco.cats[cat_id]['name']
        if cat_name in name_to_idx:
            cat_mapping[cat_id] = name_to_idx[cat_name]
    print(f"   Mapping: { {coco.cats[k]['name']: v for k, v in cat_mapping.items()} }")

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )
    print_split_stats(coco, split_stats, class_names)

    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump({
            'test_image_ids':   test_ids,
            'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
            'images_dir':       os.path.abspath(config["images_dir"]),
            'annotations_file': os.path.abspath(config["annotations_file"]),
            'num_test_images':  len(test_ids),
            'classes':          class_names,
            'all_classes':      all_cls,
            'background_label': num_classes - 1 if bg_per_img > 0 else None,
            'image_size':       IMAGE_SIZE,
            'min_crop_size':    config["min_crop_size"],
            'mode':             mode_label,
            'training_mode':    training_mode,
        }, f, indent=2)

    bright = config["aug_brightness"]
    sat    = config["aug_saturation"]
    hue    = config["aug_hue"]

    if aug_coeffs:
        class_transforms = get_class_transforms(class_names, aug_coeffs, bright, sat, hue)
        default_train_tf = get_train_transforms(bright, sat, hue)
        print("   Augmentation par classe:")
        for name in class_names:
            w = aug_coeffs.get(name, 1.0)
            print(f"      {name:<30} × {w:.2f}")
    else:
        class_transforms = None
        default_train_tf = get_train_transforms(bright, sat, hue)

    train_dataset = CropDataset(
        config["images_dir"], config["annotations_file"],
        train_ids, cat_mapping, config["min_crop_size"],
        transform=default_train_tf,
        bg_crops_per_img=bg_per_img,
        bg_iou_threshold=config["bg_iou_threshold"],
        class_transforms=class_transforms,
    )
    val_dataset = CropDataset(
        config["images_dir"], config["annotations_file"],
        val_ids, cat_mapping, config["min_crop_size"],
        transform=get_val_transforms(),
        bg_crops_per_img=bg_per_img,
        bg_iou_threshold=config["bg_iou_threshold"],
    )

    bg_train = sum(1 for s in train_dataset.samples if s['label'] == train_dataset.background_label)
    bg_val   = sum(1 for s in val_dataset.samples   if s['label'] == val_dataset.background_label)
    print(f"\n   Train crops: {len(train_dataset)} ({bg_train} bg) | Val crops: {len(val_dataset)} ({bg_val} bg)")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"],
                              shuffle=False, num_workers=0)

    print(f"\n   Chargement EfficientNet-B3 (pretrained={config['pretrained']}, attention={attention})...")
    model = build_model(num_classes,
                        pretrained=config["pretrained"],
                        freeze_backbone=True,
                        attention=attention or "none")
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {trainable:,} entraînables / {total_p:,} total")

    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )

    print("\n" + "=" * 70)
    print(f"   ENTRAÎNEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'lr':         [],
    }
    best_acc   = 0.0
    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()

        if epoch == config["freeze_epochs"] + 1:
            print(f"\n   Epoch {epoch}: Dégel du backbone — fine-tuning complet")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(),  'lr': config["lr_backbone"]},
                {'params': model.attention.parameters(), 'lr': config["learning_rate"]},
                {'params': model.pool.parameters(),      'lr': config["learning_rate"]},
                {'params': model.head.parameters(),      'lr': config["learning_rate"]},
            ], weight_decay=config["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["num_epochs"] - config["freeze_epochs"],
                eta_min=1e-7
            )

        phase = '[backbone gelé]' if epoch <= config["freeze_epochs"] else '[fine-tuning]'
        print(f"\n   Epoch [{epoch}/{config['num_epochs']}] {phase}")

        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, criterion, device)
        val_loss, val_acc, per_class_acc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device, num_classes
        )
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"   Loss: train={train_loss:.4f} val={val_loss:.4f}"
              f" | Acc: train={train_acc:.4f} val={val_acc:.4f}"
              f" | LR: {current_lr:.2e} | {format_time(time.time()-epoch_start)}")
        if per_class_acc:
            for cls_idx, acc in per_class_acc.items():
                name = all_cls[cls_idx] if cls_idx < len(all_cls) else f"class_{cls_idx}"
                print(f"      {name:<30} Acc={acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':          best_acc,
                'num_classes':      num_classes,
                'classes':          class_names,
                'all_classes':      all_cls,
                'background_label': num_classes - 1 if bg_per_img > 0 else None,
                'cat_mapping':      cat_mapping,
                'image_size':       IMAGE_SIZE,
                'mode':             mode_label,
                'training_mode':    training_mode,
            }, os.path.join(weights_dir, "best.pth"))
            print(f"   Meilleur modèle sauvegardé (val_acc: {best_acc:.4f})")

        if epoch % config["save_every"] == 0 or epoch == config["num_epochs"]:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':          val_acc,
                'num_classes':      num_classes,
                'classes':          class_names,
                'all_classes':      all_cls,
                'background_label': num_classes - 1 if bg_per_img > 0 else None,
                'cat_mapping':      cat_mapping,
                'image_size':       IMAGE_SIZE,
                'mode':             mode_label,
                'training_mode':    training_mode,
            }, os.path.join(weights_dir, "last.pth"))

        gc.collect()

    total_time = time.time() - start_time

    _, final_acc, _, all_preds, all_labels = evaluate_epoch(
        model, val_loader, criterion, device, num_classes
    )
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=all_cls, zero_division=0)
    print(f"\n   Rapport final (validation):\n{report}")

    print("\n   Copie des modèles...")
    best_model_path = os.path.join(train_dir, "best_model.pth")
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    # model_info
    model_info = {
        "model":         f"Perso_{mode_label}",
        "mode":          mode_label,
        "training_mode": training_mode,
        "best_model":    os.path.abspath(best_model_path),
        "train_dir":     os.path.abspath(train_dir),
        "classes":       class_names,
        "num_classes":   num_classes,
        "image_size":    IMAGE_SIZE,
        "best_acc":      best_acc,
        "timestamp":     timestamp,
    }
    os.makedirs(config.get("output_dir", "output"), exist_ok=True)
    info_path = os.path.join(config.get("output_dir", "output"), f"model_info_{mode_label}.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\n   model_info_{mode_label}.json -> {info_path}")

    history['time_stats'] = {
        'total_time_formatted':     format_time(total_time),
        'avg_epoch_time_formatted': format_time(total_time / config["num_epochs"]),
    }
    history['best_acc']               = best_acc
    history['confusion_matrix']       = cm.tolist()
    history['classification_report']  = report
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs_range = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs_range, history['val_loss'],   'r-', label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        if config["freeze_epochs"] < config["num_epochs"]:
            axes[0].axvline(config["freeze_epochs"], color='gray', linestyle='--', alpha=0.6)

        axes[1].plot(epochs_range, history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs_range, history['val_acc'],   'g-', label='Val')
        axes[1].set_title('Accuracy'); axes[1].set_ylim(0, 1)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs_range, history['lr'], color='orange')
        axes[2].set_title('Learning Rate'); axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(all_cls, rotation=45, ha='right')
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(all_cls)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
        ax.set_title('Matrice de Confusion (Validation)')
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"EfficientNet-B3 Perso — Mode {mode_label} [{training_mode}]\n{'='*50}\n\n")
        f.write(f"Classes:           {class_names}\n")
        f.write(f"Epochs:            {config['num_epochs']} | Batch: {config['batch_size']}\n\n")
        f.write(f"Meilleure val_acc: {best_acc:.4f}\n")
        f.write(f"Temps total:       {format_time(total_time)}\n")
        f.write(f"Chemin:            {train_dir}\n\n")
        f.write(f"Classification Report (validation):\n{report}\n")

    print("\n" + "=" * 70)
    print(f"   TERMINÉ — {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"   Meilleure val_acc: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"   Temps: {format_time(total_time)}")
    print(f"   Résultats: {train_dir}")
    print("=" * 70)

    return model, history


# =============================================================================
# OPTIMISATION OPTUNA
# =============================================================================

def _run_optimization(config, aug_coeffs, mode_label, cbam_reduction, cbam_kernel_size,
                      n_trials, n_epochs_per_trial):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    class_names = config["classes"]
    bg_per_img  = config["bg_crops_per_img"]
    num_classes = len(class_names) + (1 if bg_per_img > 0 else 0)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco = COCO(config["annotations_file"])
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    cat_mapping = {
        cat_id: name_to_idx[coco.cats[cat_id]['name']]
        for cat_id in coco.getCatIds()
        if coco.cats[cat_id]['name'] in name_to_idx
    }

    train_ids, val_ids, _, _ = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )

    bright = config["aug_brightness"]
    sat    = config["aug_saturation"]
    hue    = config["aug_hue"]

    if aug_coeffs:
        class_transforms = get_class_transforms(class_names, aug_coeffs, bright, sat, hue)
        default_train_tf = get_train_transforms(bright, sat, hue)
    else:
        class_transforms = None
        default_train_tf = get_train_transforms(bright, sat, hue)

    train_dataset = CropDataset(
        config["images_dir"], config["annotations_file"],
        train_ids, cat_mapping, config["min_crop_size"],
        transform=default_train_tf,
        bg_crops_per_img=bg_per_img,
        bg_iou_threshold=config["bg_iou_threshold"],
        class_transforms=class_transforms,
    )
    val_dataset = CropDataset(
        config["images_dir"], config["annotations_file"],
        val_ids, cat_mapping, config["min_crop_size"],
        transform=get_val_transforms(),
        bg_crops_per_img=bg_per_img,
        bg_iou_threshold=config["bg_iou_threshold"],
    )

    def objective(trial):
        lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        use_cbam     = trial.suggest_categorical("use_cbam", [True, False])

        attention = 'cbam' if use_cbam else 'none'
        model = build_model(num_classes,
                            pretrained=config["pretrained"],
                            freeze_backbone=False,
                            attention=attention)
        model.to(device)

        bs = config["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=bs,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=bs,
                                  shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_acc = 0.0
        for ep in range(n_epochs_per_trial):
            train_one_epoch(model, optimizer, train_loader, criterion, device)
            _, val_acc, _, _, _ = evaluate_epoch(model, val_loader, criterion, device, num_classes)
            best_val_acc = max(best_val_acc, val_acc)
            trial.report(val_acc, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        del model; gc.collect()
        return best_val_acc

    output_dir = OPTUNA_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(),
        study_name=f"{OPTUNA_CONFIG['study_name']}_{mode_label}",
    )
    print(f"\n   Optuna: {n_trials} trials × {n_epochs_per_trial} epochs chacun")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"\n   Meilleurs hyperparamètres: {best}")

    with open(os.path.join(output_dir, f"optuna_best_{mode_label}.json"), 'w') as f:
        json.dump({"best_params": best, "best_value": study.best_value}, f, indent=2)

    return best


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EfficientNet-B3 Perso — classification des toitures cadastrales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", default="simple",
        choices=["simple", "attention", "optimize", "dual"],
        help="Mode d'entraînement",
    )
    parser.add_argument("--aug",              nargs="*", default=[],
                        metavar="CLASS:COEFF",
                        help="Multiplicateurs d'augmentation par classe (ex: panneau_solaire:3)")
    parser.add_argument("--cbam-reduction",   type=int, default=16,
                        help="Facteur de réduction CBAM")
    parser.add_argument("--cbam-kernel-size", type=int, default=7,
                        help="Taille du kernel CBAM")
    parser.add_argument("--n-trials",         type=int, default=OPTUNA_CONFIG["n_trials"],
                        help="Nombre de trials Optuna")
    parser.add_argument("--n-epochs-trial",   type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"],
                        help="Epochs par trial Optuna")
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--classes-file",     default=None)
    args = parser.parse_args()

    mode   = args.mode
    config = _base_config()

    # Overrides CLI
    if args.images_dir:
        config["images_dir"] = args.images_dir
    if args.annotations_file:
        config["annotations_file"] = args.annotations_file
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.classes_file:
        config["classes_file"] = args.classes_file

    all_mode_classes = MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"]

    if mode == "dual":
        # Phase 1 — Nadir
        nadir_cfg            = _build_sub_config(config, "nadir")
        nadir_cfg["classes"] = load_classes(config["classes_file"], MODE_CLASSES["nadir"])
        nadir_aug            = parse_aug_coeffs(args.aug, nadir_cfg["classes"])
        print(f"\n[DUAL] Phase 1 — Nadir ({nadir_cfg['classes']})")
        _train_single(nadir_cfg, nadir_aug, "nadir", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

        # Phase 2 — Oblique
        oblique_cfg            = _build_sub_config(config, "oblique")
        oblique_cfg["classes"] = load_classes(config["classes_file"], MODE_CLASSES["oblique"])
        oblique_aug            = parse_aug_coeffs(args.aug, oblique_cfg["classes"])
        print(f"\n[DUAL] Phase 2 — Oblique ({oblique_cfg['classes']})")
        _train_single(oblique_cfg, oblique_aug, "oblique", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

    elif mode == "optimize":
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        best_params = _run_optimization(
            config, aug_coeffs, "all",
            args.cbam_reduction, args.cbam_kernel_size,
            args.n_trials, args.n_epochs_trial,
        )
        use_cbam = best_params.get("use_cbam", False)
        config["learning_rate"] = best_params.get("lr",           config["learning_rate"])
        config["weight_decay"]  = best_params.get("weight_decay", config["weight_decay"])
        cbam_r = best_params.get("cbam_reduction",   args.cbam_reduction)
        cbam_k = best_params.get("cbam_kernel_size",  args.cbam_kernel_size)
        training_mode = "attention" if use_cbam else "simple"
        _train_single(config, aug_coeffs, "all", training_mode, cbam_r, cbam_k)

    elif mode == "attention":
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "attention",
                      args.cbam_reduction, args.cbam_kernel_size)

    else:  # simple
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)


if __name__ == "__main__":
    main()
