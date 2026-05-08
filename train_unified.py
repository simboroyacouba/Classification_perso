"""
Entraînement du classificateur personnalisé — version unifiée (inspiré YOLO)

Usage:
  python train_unified.py                                  # mode all, efnet-b3, cbam
  python train_unified.py --mode nadir --backbone efficientnet_b4 --attention cbam --augment
  python train_unified.py --mode oblique --attention se --epochs 80
  python train_unified.py --freeze-epochs 0               # fine-tuning direct sans phase 1

Modes:
  nadir   → sauvegarde dans runs/classify/train/nadir/
  oblique → sauvegarde dans runs/classify/train/oblique/
  all     → sauvegarde dans runs/classify/train/all/  (dataset complet)

Backbones: efficientnet_b0, efficientnet_b3 (défaut), efficientnet_b4, resnet50
Attention: none, se, cbam (défaut)
"""

import os
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

from model import build_model, BACKBONE_IMAGE_SIZES

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = [c for c in data.get('classes', []) if c != '__background__']
    print("Classes:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:   return f"{h}h{m:02d}m{s:02d}s"
    if m:   return f"{m}m{s:02d}s"
    return f"{s}s"


def stratified_split(coco, train_split, val_split, test_split, seed=42):
    np.random.seed(seed)
    all_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_ids)
    n = len(all_ids)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test  = n - n_train - n_val
    if n_test < 1 and n > 2:
        n_test  = max(1, int(n * 0.10))
        n_train = n - n_val - n_test
    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]
    print(f"   Split: train={n_train} | val={n_val} | test={n_test}  (total={n})")
    return train_ids, val_ids, test_ids


# =============================================================================
# AUGMENTATION
# =============================================================================

def get_transforms(image_size, augment=False,
                   brightness=0.113, saturation=0.162, hue=0.05):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if augment:
        return T.Compose([
            T.ColorJitter(brightness=brightness, saturation=saturation, hue=hue),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def get_class_transforms(image_size, class_names, class_weights,
                          brightness=0.113, saturation=0.162, hue=0.05):
    """
    Retourne un dict {label_idx: transform} avec des coefficients d'augmentation
    différents par classe selon class_weights (multiplicateurs).

    Exemple : class_weights = {'panneau_solaire': 3.0, 'batiment_peint': 1.0}
    → panneau_solaire reçoit brightness*3, saturation*3, hue*3
    → batiment_peint reçoit les valeurs de base
    Classes absentes du dict reçoivent le multiplicateur 1.0.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transforms = {}
    for idx, name in enumerate(class_names):
        w = class_weights.get(name, 1.0)
        if w > 0:
            transforms[idx] = T.Compose([
                T.ColorJitter(
                    brightness=brightness * w,
                    saturation=saturation * w,
                    hue=min(hue * w, 0.5),     # hue est borné à [-0.5, 0.5]
                ),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            transforms[idx] = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
    return transforms


# =============================================================================
# DATASET — CROPS DE BBOXES COCO
# =============================================================================

def _iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def _mine_bg_crops(img_w, img_h, gt_bboxes, n=3, iou_thr=0.1, seed=None):
    rng   = np.random.RandomState(seed)
    crops = []
    for _ in range(n * 20):
        if len(crops) >= n:
            break
        scale = rng.uniform(0.05, 0.4)
        w = max(32, int(min(img_w, img_h) * scale))
        h = max(32, int(min(img_w, img_h) * scale * rng.uniform(0.7, 1.4)))
        w = min(w, img_w); h = min(h, img_h)
        x = rng.randint(0, max(1, img_w - w))
        y = rng.randint(0, max(1, img_h - h))
        if max((_iou_xywh((x,y,w,h), g) for g in gt_bboxes), default=0.0) < iou_thr:
            crops.append((x, y, w, h))
    return crops


class CropDataset(Dataset):
    def __init__(self, images_dir, annotations_file, image_ids,
                 cat_mapping, min_crop_size=10, transform=None,
                 bg_crops_per_img=0, bg_iou_threshold=0.1,
                 class_transforms=None):
        self.images_dir       = images_dir
        self.coco             = COCO(annotations_file)
        self.cat_mapping      = cat_mapping
        self.min_crop_size    = min_crop_size
        self.transform        = transform
        self.class_transforms = class_transforms  # dict {label_idx: transform}
        self.bg_crops_per_img = bg_crops_per_img
        self.bg_iou_threshold = bg_iou_threshold

        max_idx = max(cat_mapping.values()) if cat_mapping else -1
        self.num_object_classes = max_idx + 1
        self.background_label   = self.num_object_classes
        self.samples            = self._build_samples(image_ids)

    def _build_samples(self, image_ids):
        samples = []
        for img_id in image_ids:
            info     = self.coco.imgs[img_id]
            img_path = os.path.join(self.images_dir, info['file_name'])
            if not os.path.exists(img_path):
                continue
            gt_bboxes = []
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
                if ann.get('iscrowd', 0):
                    continue
                cls_idx = self.cat_mapping.get(ann['category_id'])
                if cls_idx is None:
                    continue
                x, y, w, h = ann['bbox']
                if w < self.min_crop_size or h < self.min_crop_size:
                    continue
                gt_bboxes.append((x, y, w, h))
                samples.append({'img_path': img_path, 'bbox': (x,y,w,h),
                                'label': cls_idx, 'image_id': img_id})
            if self.bg_crops_per_img > 0 and gt_bboxes:
                for bbox in _mine_bg_crops(info['width'], info['height'], gt_bboxes,
                                           n=self.bg_crops_per_img,
                                           iou_thr=self.bg_iou_threshold, seed=img_id):
                    samples.append({'img_path': img_path, 'bbox': bbox,
                                    'label': self.background_label, 'image_id': img_id})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s       = self.samples[idx]
        image   = Image.open(s['img_path']).convert("RGB")
        iw, ih  = image.size
        x, y, w, h = s['bbox']
        crop    = image.crop((max(0,int(x)), max(0,int(y)),
                              min(iw,int(x+w)), min(ih,int(y+h))))
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

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        out  = model(images)
        total_loss += criterion(out, labels).item()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n       = max(len(dataloader), 1)
    acc     = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    per_cls = {}
    for c in range(num_classes):
        mask = np.array(all_labels) == c
        if mask.sum() > 0:
            per_cls[c] = float((np.array(all_preds)[mask] == c).mean())
    return total_loss / n, acc, per_cls, all_preds, all_labels


def train_one_epoch(model, optimizer, dataloader, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out  = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    n = max(len(dataloader), 1)
    return total_loss / n, correct / max(total, 1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classificateur Perso — EfficientNet + Attention (YOLO-style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Données
    parser.add_argument("--images-dir",       default=os.getenv("DETECTION_DATASET_IMAGES_DIR",       "../dataset1/images/default"))
    parser.add_argument("--annotations-file", default=os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"))
    parser.add_argument("--classes-file",     default=os.getenv("CLASSES_FILE",                       "classes.yaml"))
    parser.add_argument("--nadir-classes",    default=os.getenv("NADIR_CLASSES_FILE",   "classes_nadir.yaml"))
    parser.add_argument("--oblique-classes",  default=os.getenv("OBLIQUE_CLASSES_FILE", "classes_oblique.yaml"))
    parser.add_argument("--nadir-annotations",   default=os.getenv("NADIR_ANNOTATIONS_FILE",   None),
                        help="Fichier annotations spécifique au mode nadir (optionnel)")
    parser.add_argument("--oblique-annotations", default=os.getenv("OBLIQUE_ANNOTATIONS_FILE", None),
                        help="Fichier annotations spécifique au mode oblique (optionnel)")

    # Mode
    parser.add_argument("--mode", choices=["nadir", "oblique", "all"], default="all",
                        help="nadir / oblique / all")

    # Modèle
    parser.add_argument("--backbone",   default=os.getenv("BACKBONE", "efficientnet_b3"),
                        choices=["efficientnet_b0", "efficientnet_b3", "efficientnet_b4", "resnet50"])
    parser.add_argument("--attention",  default=os.getenv("ATTENTION", "cbam"),
                        choices=["none", "se", "cbam"])
    parser.add_argument("--img-size",   type=int, default=None,
                        help="Taille image (défaut: taille naturelle du backbone)")
    parser.add_argument("--dropout",    type=float, default=float(os.getenv("DROPOUT", "0.4")))
    parser.add_argument("--no-pretrained", action="store_true")

    # Entraînement
    parser.add_argument("--epochs",        type=int,   default=int(os.getenv("NUM_EPOCHS",    "60")))
    parser.add_argument("--freeze-epochs", type=int,   default=int(os.getenv("FREEZE_EPOCHS", "10")),
                        help="Epochs avec backbone gelé (phase 1). 0 = pas de phase gelée.")
    parser.add_argument("--batch-size",    type=int,   default=int(os.getenv("BATCH_SIZE",    "16")))
    parser.add_argument("--lr",            type=float, default=float(os.getenv("LEARNING_RATE", "1e-3")))
    parser.add_argument("--lr-backbone",   type=float, default=float(os.getenv("LR_BACKBONE",   "1e-5")),
                        help="LR du backbone lors du fine-tuning complet (phase 2)")
    parser.add_argument("--weight-decay",  type=float, default=float(os.getenv("WEIGHT_DECAY",  "1e-4")))
    parser.add_argument("--label-smoothing", type=float, default=float(os.getenv("LABEL_SMOOTHING", "0.1")))
    parser.add_argument("--augment",       action="store_true", default=os.getenv("AUGMENT","0")=="1",
                        help="Active l'augmentation couleur")
    parser.add_argument("--aug-brightness", type=float, default=float(os.getenv("AUG_BRIGHTNESS", "0.113")),
                        help="Variation de luminosite (ex: 0.113 = ±11.3%%)")
    parser.add_argument("--aug-saturation", type=float, default=float(os.getenv("AUG_SATURATION", "0.162")),
                        help="Variation de saturation (ex: 0.162 = ±16.2%%)")
    parser.add_argument("--aug-hue",        type=float, default=float(os.getenv("AUG_HUE",        "0.05")),
                        help="Variation de teinte (ex: 0.05 = ±5%%)")
    parser.add_argument("--aug-class-weights", nargs="*", default=None, metavar="CLASSE:MULT",
                        help="Multiplicateur d'augmentation par classe. "
                             "Ex: panneau_solaire:3.0 batiment_peint:1.0 "
                             "(1.0 = valeurs de base, 0.0 = pas d'augmentation)")
    parser.add_argument("--amp",           action="store_true", default=os.getenv("USE_AMP","0")=="1",
                        help="Mixed precision (GPU uniquement)")
    parser.add_argument("--bg-crops",      type=int,   default=int(os.getenv("BG_CROPS_PER_IMG", "3")),
                        help="Crops background par image (0 = désactivé)")
    parser.add_argument("--min-crop-size", type=int,   default=int(os.getenv("MIN_CROP_SIZE", "10")))
    parser.add_argument("--train-split",   type=float, default=float(os.getenv("TRAIN_SPLIT", "0.70")))
    parser.add_argument("--val-split",     type=float, default=float(os.getenv("VAL_SPLIT",   "0.20")))
    parser.add_argument("--test-split",    type=float, default=float(os.getenv("TEST_SPLIT",  "0.10")))
    parser.add_argument("--save-every",    type=int,   default=int(os.getenv("SAVE_EVERY",    "5")))
    parser.add_argument("--output-dir",    default=os.getenv("OUTPUT_DIR", "./runs/classify/train"))
    args = parser.parse_args()

    # Résolution des paramètres mode-dépendants
    if args.mode == "nadir":
        classes_file = args.nadir_classes if os.path.exists(args.nadir_classes) else args.classes_file
        ann_file     = args.nadir_annotations or args.annotations_file
    elif args.mode == "oblique":
        classes_file = args.oblique_classes if os.path.exists(args.oblique_classes) else args.classes_file
        ann_file     = args.oblique_annotations or args.annotations_file
    else:
        classes_file = args.classes_file
        ann_file     = args.annotations_file

    image_size = args.img_size or BACKBONE_IMAGE_SIZES.get(args.backbone, 299)

    # -------------------------------------------------------------------------
    print("=" * 70)
    print(f"   Classificateur Perso — Mode: {args.mode.upper()}")
    print("=" * 70)

    class_names = load_classes(classes_file)
    bg_per_img  = args.bg_crops
    num_classes = len(class_names) + (1 if bg_per_img > 0 else 0)
    all_classes = class_names + (['background'] if bg_per_img > 0 else [])

    print(f"   Backbone:    {args.backbone} ({image_size}px)")
    print(f"   Attention:   {args.attention}")
    print(f"   Augment:     {args.augment}")
    print(f"   AMP:         {args.amp}")
    print(f"   Classes:     {num_classes}  ({', '.join(all_classes)})")
    print(f"   Epochs:      {args.epochs}  (phase1={args.freeze_epochs} gelé)")
    print(f"   Batch:       {args.batch_size}  LR={args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    # Dossier de sortie
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name    = f"perso_{args.mode}_{args.backbone}_{args.attention}_{timestamp}"
    out_dir     = os.path.join(args.output_dir, args.mode, run_name)
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Dataset
    coco        = COCO(ann_file)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    cat_mapping = {}
    for cat_id in coco.getCatIds():
        cat_name = coco.cats[cat_id]['name']
        if cat_name in name_to_idx:
            cat_mapping[cat_id] = name_to_idx[cat_name]
    print(f"   Mapping: { {coco.cats[k]['name']: v for k, v in cat_mapping.items()} }")

    train_ids, val_ids, test_ids = stratified_split(
        coco, args.train_split, args.val_split, args.test_split)

    # Sauvegarde test_info
    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       os.path.abspath(args.images_dir),
        'annotations_file': os.path.abspath(ann_file),
        'classes':          class_names,
        'all_classes':      all_classes,
        'background_label': num_classes - 1 if bg_per_img > 0 else None,
        'image_size':       image_size,
        'backbone':         args.backbone,
        'attention':        args.attention,
        'mode':             args.mode,
    }
    with open(os.path.join(out_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    # Parse des poids d'augmentation par classe
    aug_class_weights = {}
    if args.aug_class_weights:
        for item in args.aug_class_weights:
            try:
                name, mult = item.rsplit(":", 1)
                aug_class_weights[name.strip()] = float(mult)
            except ValueError:
                print(f"   Format invalide pour --aug-class-weights: '{item}' (attendu: nom:multiplicateur)")

    val_tf = get_transforms(image_size, augment=False)

    if args.augment and aug_class_weights:
        # Augmentation différente par classe
        class_transforms = get_class_transforms(
            image_size, class_names, aug_class_weights,
            brightness=args.aug_brightness,
            saturation=args.aug_saturation,
            hue=args.aug_hue,
        )
        default_tf = get_transforms(image_size, augment=True,
                                    brightness=args.aug_brightness,
                                    saturation=args.aug_saturation,
                                    hue=args.aug_hue)
        print("   Augmentation par classe:")
        for name in class_names:
            w = aug_class_weights.get(name, 1.0)
            print(f"      {name:<30} × {w:.2f}  "
                  f"(bright={args.aug_brightness*w:.3f} "
                  f"sat={args.aug_saturation*w:.3f} "
                  f"hue={min(args.aug_hue*w, 0.5):.3f})")
        train_tf = default_tf
    else:
        class_transforms = None
        train_tf = get_transforms(image_size, augment=args.augment,
                                  brightness=args.aug_brightness,
                                  saturation=args.aug_saturation,
                                  hue=args.aug_hue)

    train_ds = CropDataset(args.images_dir, ann_file, train_ids, cat_mapping,
                           args.min_crop_size, train_tf, bg_per_img, 0.1,
                           class_transforms=class_transforms)
    val_ds   = CropDataset(args.images_dir, ann_file, val_ids,   cat_mapping,
                           args.min_crop_size, val_tf,   bg_per_img, 0.1)

    bg_tr = sum(1 for s in train_ds.samples if s['label'] == train_ds.background_label)
    bg_vl = sum(1 for s in val_ds.samples   if s['label'] == val_ds.background_label)
    print(f"   Train: {len(train_ds)} crops ({bg_tr} bg) | Val: {len(val_ds)} ({bg_vl} bg)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Modèle — phase 1 backbone gelé si freeze_epochs > 0
    pretrained    = not args.no_pretrained
    freeze_phase1 = args.freeze_epochs > 0
    model = build_model(num_classes, backbone=args.backbone, attention=args.attention,
                        pretrained=pretrained, freeze_backbone=freeze_phase1,
                        dropout=args.dropout)
    model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"   Params: {n_trainable/1e6:.2f}M entraînables / {n_total/1e6:.2f}M total")
    if freeze_phase1:
        print(f"   Phase 1: backbone gelé ({args.freeze_epochs} epochs)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"   ENTRAINEMENT")
    print("=" * 70)
    print(f"   {'Epoch':>6}  {'train_loss':>10}  {'train_acc':>9}  "
          f"{'val_loss':>9}  {'val_acc':>8}  {'LR':>9}  {'temps':>8}")
    print(f"   {'─'*65}")

    history    = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'lr':[]}
    best_acc   = 0.0
    best_path  = os.path.join(weights_dir, "best.pth")
    start_time = time.time()

    def _save_ckpt(path, epoch, val_acc_val):
        torch.save({
            'epoch':            epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc':          val_acc_val,
            'num_classes':      num_classes,
            'classes':          class_names,
            'all_classes':      all_classes,
            'background_label': num_classes - 1 if bg_per_img > 0 else None,
            'cat_mapping':      cat_mapping,
            'image_size':       image_size,
            'backbone':         args.backbone,
            'attention':        args.attention,
            'mode':             args.mode,
        }, path)

    for epoch in range(1, args.epochs + 1):
        epoch_t = time.time()

        # Passage phase 2 : dégel du backbone
        if freeze_phase1 and epoch == args.freeze_epochs + 1:
            print(f"\n   [Phase 2] Dégel backbone — fine-tuning complet (LR backbone={args.lr_backbone})")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(),  'lr': args.lr_backbone},
                {'params': model.attention.parameters(), 'lr': args.lr},
                {'params': model.pool.parameters(),      'lr': args.lr},
                {'params': model.head.parameters(),      'lr': args.lr},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.freeze_epochs,
                eta_min=1e-7,
            )

        tr_loss, tr_acc = train_one_epoch(model, optimizer, train_loader,
                                          criterion, device, scaler)
        vl_loss, vl_acc, per_cls, _, _ = evaluate_epoch(
            model, val_loader, criterion, device, num_classes)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        history['lr'].append(current_lr)

        phase = "P1" if epoch <= args.freeze_epochs else "P2"
        print(f"   {epoch:>5}/{args.epochs}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{vl_loss:>9.4f}  {vl_acc:>8.4f}  {current_lr:>9.2e}  "
              f"{format_time(time.time()-epoch_t):>8}  [{phase}]")

        if vl_acc > best_acc:
            best_acc = vl_acc
            _save_ckpt(best_path, epoch, best_acc)
            print(f"   >>> best model  val_acc={best_acc:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_ckpt(os.path.join(weights_dir, "last.pth"), epoch, vl_acc)

        gc.collect()

    total_time = time.time() - start_time

    # Rapport final
    shutil.copy2(best_path, os.path.join(out_dir, "best_model.pth"))
    shutil.copy2(os.path.join(weights_dir, "last.pth"), os.path.join(out_dir, "final_model.pth"))

    _, _, _, all_preds, all_labels = evaluate_epoch(
        model, val_loader, criterion, device, num_classes)
    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=all_classes, zero_division=0)
    print(f"\n   Rapport val:\n{report}")

    history.update({'best_acc': best_acc,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report,
                    'args': vars(args)})
    with open(os.path.join(out_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Courbes
    if history['train_loss']:
        ep = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(ep, history['train_loss'], label='Train')
        axes[0].plot(ep, history['val_loss'],   label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        if args.freeze_epochs > 0 and args.freeze_epochs < args.epochs:
            axes[0].axvline(args.freeze_epochs, color='gray', linestyle='--', label='Dégel')

        axes[1].plot(ep, history['train_acc'], label='Train')
        axes[1].plot(ep, history['val_acc'],   label='Val')
        axes[1].set_title('Accuracy'); axes[1].set_ylim(0, 1)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(ep, history['lr'], color='orange')
        axes[2].set_title('LR'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
        plt.close()

        # Matrice de confusion
        fig, ax = plt.subplots(figsize=(max(6, len(all_classes)), max(5, len(all_classes)-1)))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(all_classes))); ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.set_yticks(range(len(all_classes))); ax.set_yticklabels(all_classes)
        for i in range(len(all_classes)):
            for j in range(len(all_classes)):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
        ax.set_title('Matrice de Confusion (Validation)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    print("\n" + "=" * 70)
    print(f"   TERMINE — {args.mode.upper()}")
    print("=" * 70)
    print(f"   Meilleure val_acc : {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"   Temps             : {format_time(total_time)}")
    print(f"   Resultats         : {out_dir}")
    print("=" * 70)

    return best_acc, out_dir


if __name__ == "__main__":
    main()
