"""
Évaluation du classificateur personnalisé - Toitures cadastrales
Évaluation sur le TEST SET (10% du dataset)
Configuration: test_info.json généré par train.py
"""

import os
import json
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from model import build_model
from train import CropDataset, get_val_transforms, format_time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


CONFIG = {
    "model_path":   os.getenv("MODEL_PATH", None),
    "output_dir":   os.getenv("EVALUATION_DIR", "./evaluation"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "batch_size":   int(os.getenv("BATCH_SIZE", "16")),
}


def load_model(model_path, device):
    print(f"🧠 Chargement du modèle: {model_path}")
    checkpoint  = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    classes     = checkpoint['classes']
    cat_mapping = checkpoint['cat_mapping']
    image_size  = checkpoint.get('image_size', 299)

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"   ✅ Époque {checkpoint['epoch']} | val_acc={checkpoint['val_acc']:.4f}")
    print(f"   Classes: {classes}")
    return model, classes, {int(k): v for k, v in cat_mapping.items()}, image_size


@torch.no_grad()
def run_evaluation(model, dataloader, class_names, device):
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))


def evaluate():
    model_path = CONFIG["model_path"]
    if not model_path:
        raise ValueError("MODEL_PATH non défini dans .env ou variable d'environnement")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names, cat_mapping, image_size = load_model(model_path, device)
    num_classes = len(class_names)

    # Récupérer test_info.json depuis le dossier du modèle
    model_dir = os.path.dirname(model_path)
    # Chercher test_info.json dans le dossier parent (train_dir)
    for candidate in [model_dir, os.path.dirname(model_dir)]:
        test_info_path = os.path.join(candidate, "test_info.json")
        if os.path.exists(test_info_path):
            break
    else:
        raise FileNotFoundError("test_info.json introuvable. Relancer train.py d'abord.")

    with open(test_info_path) as f:
        test_info = json.load(f)

    images_dir       = test_info['images_dir']
    annotations_file = test_info['annotations_file']
    test_image_ids   = test_info['test_image_ids']
    min_crop_size    = test_info.get('min_crop_size', 10)

    print(f"\n📂 Test set: {len(test_image_ids)} images")

    test_dataset = CropDataset(
        images_dir, annotations_file,
        test_image_ids, cat_mapping, min_crop_size,
        transform=get_val_transforms()
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, num_workers=0)

    print(f"   Total crops test: {len(test_dataset)}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG["output_dir"], f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n⏳ Évaluation en cours...")
    t0 = time.time()
    labels, preds, probs = run_evaluation(model, test_loader, class_names, device)
    elapsed = time.time() - t0

    acc       = accuracy_score(labels, preds)
    f1_macro  = f1_score(labels, preds, average='macro',    zero_division=0)
    f1_weight = f1_score(labels, preds, average='weighted', zero_division=0)
    prec      = precision_score(labels, preds, average='macro', zero_division=0)
    rec       = recall_score(labels, preds, average='macro',    zero_division=0)
    cm        = confusion_matrix(labels, preds)
    report    = classification_report(labels, preds, target_names=class_names, zero_division=0)

    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS")
    print("=" * 70)
    print(f"   Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1 macro:        {f1_macro:.4f}")
    print(f"   F1 weighted:     {f1_weight:.4f}")
    print(f"   Précision macro: {prec:.4f}")
    print(f"   Rappel macro:    {rec:.4f}")
    print(f"   Temps:           {format_time(elapsed)} ({len(test_dataset)} crops)")
    print(f"\n{report}")

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
    ax.set_title('Matrice de Confusion — Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Sauvegarde JSON
    results = {
        'model_path':    model_path,
        'classes':       class_names,
        'num_test_crops': int(len(test_dataset)),
        'metrics': {
            'accuracy':         float(acc),
            'f1_macro':         float(f1_macro),
            'f1_weighted':      float(f1_weight),
            'precision_macro':  float(prec),
            'recall_macro':     float(rec),
        },
        'confusion_matrix':        cm.tolist(),
        'classification_report':   report,
        'inference_time_seconds':  float(elapsed),
    }
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   📁 Résultats: {output_dir}")
    return results


if __name__ == "__main__":
    evaluate()
