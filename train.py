"""
Entraînement du classificateur personnalisé - Toitures cadastrales
Dataset : Images aériennes annotées avec CVAT (format COCO)
          → chaque bbox est cropée et classifiée (pas de détection)
Classes : Chargées depuis classes.yaml
Config  : Chargée depuis .env

Architecture : EfficientNet-B3 + ResidualAttentionBlock + CustomPoolingConcatenate
Stratégie    : Phase 1 backbone gelé (head only) → Phase 2 fine-tuning complet
"""

import os
import json
import yaml
import shutil
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
# CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', [])
    # Exclure __background__ : le classificateur n'en a pas besoin
    classes = [c for c in classes if c != '__background__']
    print(f"📋 Classes chargées depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "images_dir":       os.getenv("DETECTION_DATASET_IMAGES_DIR",       "../dataset1/images/default"),
    "annotations_file": os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "output_dir":       os.getenv("OUTPUT_DIR",   "./output"),
    "classes_file":     os.getenv("CLASSES_FILE", "classes.yaml"),
    "classes":          None,

    # Hyperparamètres
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

    # Phases de fine-tuning
    "freeze_epochs":    int(os.getenv("FREEZE_EPOCHS", "10")),
    "lr_backbone":      float(os.getenv("LR_BACKBONE",  "1e-5")),

    # Background mining : crops "vides" pour apprendre à rejeter le fond
    "bg_crops_per_img": int(os.getenv("BG_CROPS_PER_IMG", "3")),   # crops bg par image
    "bg_iou_threshold": float(os.getenv("BG_IOU_THRESHOLD", "0.1")),  # IoU max avec GT
}

IMAGE_SIZE = 299  # Même taille que le modèle original


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

    print(f"\n   📊 Split des IMAGES (total: {n_total}):")
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
    print("\n   📊 Distribution des crops par classe (split 70/20/10):")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = "⚠️" if val == 0 or test == 0 else "✅"
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8} {ok}")
    print(f"   {'-'*70}")


# =============================================================================
# DATASET — CROPS DE BBOXES COCO
# =============================================================================

def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _iou_xywh(box_a, box_b):
    """IoU entre deux boîtes [x, y, w, h]."""
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
    """
    Génère jusqu'à n crops aléatoires dont l'IoU avec tous les GT < iou_threshold.
    Retourne une liste de (x, y, w, h).
    """
    rng   = np.random.RandomState(seed)
    crops = []
    for _ in range(n * 20):           # essais max
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
      → permet au modèle de rejeter le fond pendant le sliding window

    bg_crops_per_img=0 désactive le background mining (mode évaluation).
    """

    BACKGROUND_LABEL = None  # défini dynamiquement à num_object_classes

    def __init__(self, images_dir, annotations_file, image_ids,
                 cat_mapping, min_crop_size=10, transform=None,
                 bg_crops_per_img=0, bg_iou_threshold=0.1):
        self.images_dir       = images_dir
        self.coco             = COCO(annotations_file)
        self.cat_mapping      = cat_mapping
        self.min_crop_size    = min_crop_size
        self.transform        = transform
        self.bg_crops_per_img = bg_crops_per_img
        self.bg_iou_threshold = bg_iou_threshold

        max_idx = max(cat_mapping.values()) if cat_mapping else -1
        if max_idx < 0:
            raise ValueError("cat_mapping vide — aucune catégorie COCO ne correspond aux classes yaml.")
        self.num_object_classes = max_idx + 1
        self.background_label   = self.num_object_classes  # dernier indice
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

            # Background mining
            if self.bg_crops_per_img > 0 and gt_bboxes:
                img_w = img_info['width']
                img_h = img_info['height']
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
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + w))
        y2 = min(img_h, int(y + h))
        crop = image.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)
        return crop, s['label']


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

    n       = max(len(dataloader), 1)
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
    total_loss  = 0.0
    correct     = 0
    total       = 0

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
# MAIN
# =============================================================================

def train_perso():
    CONFIG["classes"] = load_classes(CONFIG["classes_file"])
    class_names       = CONFIG["classes"]
    bg_per_img        = CONFIG["bg_crops_per_img"]
    # +1 classe background si le mining est activé
    num_classes       = len(class_names) + (1 if bg_per_img > 0 else 0)
    all_class_names   = class_names + (['background'] if bg_per_img > 0 else [])

    print("=" * 70)
    print("   EfficientNet-B3 Perso — Classification des Toitures")
    print("=" * 70)
    print(f"\n📋 CONFIG (.env)")
    print(f"   Images:      {CONFIG['images_dir']}")
    print(f"   Annotations: {CONFIG['annotations_file']}")
    print(f"   Classes:     {num_classes} ({', '.join(all_class_names)})")
    print(f"   Background:  {bg_per_img} crops/image (IoU < {CONFIG['bg_iou_threshold']})")
    print(f"   Epochs:      {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"   Phase 1:     {CONFIG['freeze_epochs']} epochs backbone gelé")
    print(f"   Image size:  {IMAGE_SIZE}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join("runs", "classify", "train", f"perso_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Split
    coco = COCO(CONFIG["annotations_file"])

    # Mapping par NOM de classe (robuste si le COCO contient des catégories extra
    # ou dans un ordre différent — évite les labels hors-range)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    cat_mapping = {}
    for cat_id in coco.getCatIds():
        cat_name = coco.cats[cat_id]['name']
        if cat_name in name_to_idx:
            cat_mapping[cat_id] = name_to_idx[cat_name]
    print(f"   Mapping COCO → classe: { {coco.cats[k]['name']: v for k, v in cat_mapping.items()} }")

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, CONFIG["train_split"], CONFIG["val_split"], CONFIG["test_split"], seed=42
    )
    print_split_stats(coco, split_stats, class_names)

    # Sauvegarde test_info pour evaluate.py
    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       os.path.abspath(CONFIG["images_dir"]),
        'annotations_file': os.path.abspath(CONFIG["annotations_file"]),
        'num_test_images':  len(test_ids),
        'classes':          class_names,       # classes objet uniquement
        'all_classes':      all_class_names,   # inclut background si activé
        'background_label': num_classes - 1 if bg_per_img > 0 else None,
        'image_size':       IMAGE_SIZE,
        'min_crop_size':    CONFIG["min_crop_size"],
    }
    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    # Datasets
    train_dataset = CropDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        train_ids, cat_mapping, CONFIG["min_crop_size"],
        transform=get_train_transforms(),
        bg_crops_per_img=bg_per_img,
        bg_iou_threshold=CONFIG["bg_iou_threshold"],
    )
    val_dataset = CropDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        val_ids, cat_mapping, CONFIG["min_crop_size"],
        transform=get_val_transforms(),
        bg_crops_per_img=bg_per_img,
        bg_iou_threshold=CONFIG["bg_iou_threshold"],
    )

    bg_train = sum(1 for s in train_dataset.samples if s['label'] == train_dataset.background_label)
    bg_val   = sum(1 for s in val_dataset.samples   if s['label'] == val_dataset.background_label)
    print(f"\n   Train crops: {len(train_dataset)} ({bg_train} bg) | Val crops: {len(val_dataset)} ({bg_val} bg)")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    # Modèle — Phase 1 : backbone gelé
    print(f"\n🧠 Chargement EfficientNet-B3 (pretrained={CONFIG['pretrained']})...")
    model = build_model(num_classes, pretrained=CONFIG["pretrained"], freeze_backbone=True)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres: {trainable_params:,} entraînables / {total_params:,} total")

    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )

    print("\n" + "=" * 70)
    print(f"   🚀 ENTRAÎNEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'lr':         [],
    }
    best_acc   = 0.0
    start_time = time.time()

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start = time.time()

        # Phase 2 : dégel du backbone à partir de freeze_epochs
        if epoch == CONFIG["freeze_epochs"] + 1:
            print(f"\n🔓 Epoch {epoch}: Dégel du backbone — fine-tuning complet")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(),  'lr': CONFIG["lr_backbone"]},
                {'params': model.attention.parameters(), 'lr': CONFIG["learning_rate"]},
                {'params': model.pool.parameters(),      'lr': CONFIG["learning_rate"]},
                {'params': model.head.parameters(),      'lr': CONFIG["learning_rate"]},
            ], weight_decay=CONFIG["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["num_epochs"] - CONFIG["freeze_epochs"],
                eta_min=1e-7
            )

        print(f"\n📅 Epoch [{epoch}/{CONFIG['num_epochs']}]"
              f"{'  [backbone gelé]' if epoch <= CONFIG['freeze_epochs'] else '  [fine-tuning]'}")

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
              f" | LR: {current_lr:.2e} | ⏱️ {format_time(time.time()-epoch_start)}")
        if per_class_acc:
            for cls_idx, acc in per_class_acc.items():
                name = all_class_names[cls_idx] if cls_idx < len(all_class_names) else f"class_{cls_idx}"
                print(f"      {name:<30} Acc={acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':    best_acc,
                'num_classes':      num_classes,
                'classes':          class_names,
                'all_classes':      all_class_names,
                'background_label': num_classes - 1 if bg_per_img > 0 else None,
                'cat_mapping':      cat_mapping,
                'image_size':       IMAGE_SIZE,
            }, os.path.join(weights_dir, "best.pth"))
            print(f"   💾 Meilleur modèle sauvegardé (val_acc: {best_acc:.4f})")

        if epoch % CONFIG["save_every"] == 0 or epoch == CONFIG["num_epochs"]:
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':    val_acc,
                'num_classes':      num_classes,
                'classes':          class_names,
                'all_classes':      all_class_names,
                'background_label': num_classes - 1 if bg_per_img > 0 else None,
                'cat_mapping':      cat_mapping,
                'image_size':       IMAGE_SIZE,
            }, os.path.join(weights_dir, "last.pth"))

        gc.collect()

    total_time = time.time() - start_time

    # Évaluation finale sur val avec rapport complet
    _, final_acc, _, all_preds, all_labels = evaluate_epoch(
        model, val_loader, criterion, device, num_classes
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=all_class_names, zero_division=0)
    print(f"\n📊 Rapport final (validation):\n{report}")

    # Copies des checkpoints
    print("\n📦 Copie des modèles...")
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   ✅ {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    # Sauvegarde history + rapport
    history['time_stats'] = {
        'total_time_formatted':     format_time(total_time),
        'avg_epoch_time_formatted': format_time(total_time / CONFIG["num_epochs"]),
    }
    history['config']    = CONFIG
    history['best_acc']  = best_acc
    history['confusion_matrix'] = cm.tolist()
    history['classification_report'] = report
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Courbes d'entraînement
    if history['train_loss']:
        epochs_range = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs_range, history['val_loss'],   'r-', label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        if CONFIG["freeze_epochs"] < CONFIG["num_epochs"]:
            axes[0].axvline(CONFIG["freeze_epochs"], color='gray', linestyle='--', label='Dégel')

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

        # Matrice de confusion
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
        ax.set_title('Matrice de Confusion (Validation)')
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    # Rapport texte
    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"EfficientNet-B3 Perso — Rapport\n{'='*50}\n\n")
        f.write(f"Classes:         {class_names}\n")
        f.write(f"Epochs:          {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}\n\n")
        f.write(f"Meilleure val_acc: {best_acc:.4f}\n")
        f.write(f"Temps total:       {format_time(total_time)}\n")
        f.write(f"Chemin:            {train_dir}\n\n")
        f.write(f"Classification Report (validation):\n{report}\n")

    print("\n" + "=" * 70)
    print("   🎉 TERMINÉ")
    print("=" * 70)
    print(f"   Meilleure val_acc: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"   ⏱️  Temps: {format_time(total_time)}")
    print(f"   📁 Résultats: {train_dir}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    train_perso()
