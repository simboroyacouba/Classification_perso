"""
Recherche d'hyperparamètres — Classificateur Perso
Méthodes: grid search, random search
Chaque trial lance train_unified.py en sous-processus et rapporte val_acc + temps.

Usage:
  python train_search.py --method random --n-trials 10 --epochs 20
  python train_search.py --method grid   --epochs 15 --mode nadir
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# ESPACE DE RECHERCHE
# Modifiez ici les valeurs à tester.
# =============================================================================

SEARCH_SPACE = {
    "lr":            [1e-4, 5e-4, 1e-3],
    "batch_size":    [8, 16, 32],
    "weight_decay":  [1e-5, 1e-4, 5e-4],
    "dropout":       [0.3, 0.4, 0.5],
    "freeze_epochs": [0, 10, 20],
    "attention":     ["none", "se", "cbam"],
    "augment":       [False, True],
    "backbone":      ["efficientnet_b3", "efficientnet_b4"],
}


# =============================================================================
# MÉTHODES DE RECHERCHE
# =============================================================================

def grid_search(space):
    keys   = list(space.keys())
    values = list(space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def random_search(space, n, seed=42):
    rng    = random.Random(seed)
    trials = []
    for _ in range(n):
        trials.append({k: rng.choice(v) for k, v in space.items()})
    return trials


# =============================================================================
# CONSTRUCTION DE LA COMMANDE
# =============================================================================

def build_cmd(params, base_args):
    cmd = [sys.executable, "train_unified.py"] + base_args
    cmd += ["--lr",            str(params["lr"])]
    cmd += ["--batch-size",    str(int(params["batch_size"]))]
    cmd += ["--weight-decay",  str(params["weight_decay"])]
    cmd += ["--dropout",       str(params["dropout"])]
    cmd += ["--freeze-epochs", str(int(params["freeze_epochs"]))]
    cmd += ["--attention",     params["attention"]]
    cmd += ["--backbone",      params["backbone"]]
    if params.get("augment"):
        cmd += ["--augment"]
    return cmd


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:   return f"{h}h{m:02d}m{s:02d}s"
    if m:   return f"{m}m{s:02d}s"
    return f"{s}s"


# =============================================================================
# LECTURE DES RÉSULTATS D'UN TRIAL
# =============================================================================

def find_latest_run(runs_dir, mode, created_after):
    """Retourne le répertoire de run créé après created_after."""
    search_dir = os.path.join(runs_dir, mode)
    if not os.path.exists(search_dir):
        return None
    best_dir, best_t = None, 0
    for d in Path(search_dir).iterdir():
        if not d.is_dir():
            continue
        t = d.stat().st_ctime
        if t >= created_after and t > best_t:
            best_t, best_dir = t, d
    return best_dir


def read_best_acc(run_dir):
    if run_dir is None:
        return None
    h_path = Path(run_dir) / "history.json"
    if not h_path.exists():
        return None
    with open(h_path) as f:
        h = json.load(f)
    return h.get("best_acc")


# =============================================================================
# LANCEMENT D'UN TRIAL
# =============================================================================

def run_trial(params, base_args, trial_idx, total, runs_dir, mode):
    cmd = build_cmd(params, base_args)

    print(f"\n{'─'*60}")
    print(f"  Trial [{trial_idx}/{total}] — {datetime.now().strftime('%H:%M:%S')}")
    for k, v in params.items():
        print(f"    {k:<20} = {v}")
    print(f"  Commande: {' '.join(cmd)}")
    print(f"{'─'*60}")

    created_after = time.time() - 1
    t0   = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    run_dir = find_latest_run(runs_dir, mode, created_after)
    acc     = read_best_acc(run_dir)

    status = "OK" if proc.returncode == 0 else f"ERREUR (code {proc.returncode})"
    print(f"\n  [{status}] val_acc = {acc:.4f if acc is not None else '?'}"
          f"  |  temps = {format_time(elapsed)}")

    return {
        "trial":      trial_idx,
        "params":     params,
        "val_acc":    acc,
        "time_s":     round(elapsed, 1),
        "time_fmt":   format_time(elapsed),
        "returncode": proc.returncode,
        "run_dir":    str(run_dir) if run_dir else None,
    }


# =============================================================================
# RAPPORT FINAL
# =============================================================================

def print_report(results, method, total_time):
    valid = [r for r in results if r["val_acc"] is not None]
    valid.sort(key=lambda r: r["val_acc"], reverse=True)

    print("\n" + "=" * 70)
    print(f"  RÉSULTATS — {method.upper()} SEARCH ({len(results)} trials)")
    print("=" * 70)
    print(f"  {'#':<4} {'val_acc':>8} {'temps':>9}  hyperparamètres")
    print(f"  {'─'*65}")

    for rank, r in enumerate(valid, 1):
        p  = r["params"]
        hp = (f"lr={p['lr']}  bs={int(p['batch_size'])}"
              f"  wd={p['weight_decay']}  drop={p['dropout']}"
              f"  freeze={int(p['freeze_epochs'])}"
              f"  {p['backbone']}  attn={p['attention']}")
        if p.get("augment"): hp += "  [aug]"
        print(f"  {rank:<4} {r['val_acc']:>8.4f} {r['time_fmt']:>9}  {hp}")

    print(f"\n  Temps total : {format_time(total_time)}")
    print(f"  Temps moyen : {format_time(total_time / max(len(results),1))}/trial")

    if valid:
        best = valid[0]
        print(f"\n  Meilleurs hyperparamètres (val_acc = {best['val_acc']:.4f}):")
        for k, v in best["params"].items():
            print(f"    {k:<20} = {v}")
        print(f"\n  Pour entraîner le modèle final avec ces paramètres :")
        cmd_args = [a for a in build_cmd(best["params"], [])
                    if a not in (sys.executable, "train_unified.py")]
        print(f"    python train_unified.py {' '.join(cmd_args)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recherche d'hyperparamètres — Classificateur Perso",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method",   choices=["grid", "random"], default="random")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Nombre d'essais (random search)")
    parser.add_argument("--epochs",   type=int, default=int(os.getenv("SEARCH_EPOCHS", "20")),
                        help="Epochs par trial")
    parser.add_argument("--mode",     choices=["nadir", "oblique", "all"], default="all")
    parser.add_argument("--output",   default="./search_results")
    parser.add_argument("--runs-dir", default=os.getenv("OUTPUT_DIR", "./runs/classify/train"))
    parser.add_argument("--seed",     type=int, default=42)
    # Transmis à train_unified.py
    parser.add_argument("--images-dir",       default=os.getenv("DETECTION_DATASET_IMAGES_DIR",       None))
    parser.add_argument("--annotations-file", default=os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", None))
    parser.add_argument("--classes-file",     default=os.getenv("CLASSES_FILE",                       None))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.method == "grid":
        trials = grid_search(SEARCH_SPACE)
        print(f"Grid search : {len(trials)} combinaisons × {args.epochs} epochs")
    else:
        trials = random_search(SEARCH_SPACE, args.n_trials, seed=args.seed)
        print(f"Random search : {args.n_trials} essais × {args.epochs} epochs")

    base_args = ["--epochs", str(args.epochs), "--mode", args.mode]
    if args.images_dir:
        base_args += ["--images-dir",       args.images_dir]
    if args.annotations_file:
        base_args += ["--annotations-file", args.annotations_file]
    if args.classes_file:
        base_args += ["--classes-file",     args.classes_file]

    print(f"Mode: {args.mode} | Méthode: {args.method} | Résultats: {args.output}")

    results   = []
    start_all = time.time()

    for i, params in enumerate(trials, 1):
        result = run_trial(params, base_args, i, len(trials), args.runs_dir, args.mode)
        results.append(result)

        out_path = os.path.join(args.output, "search_results.json")
        with open(out_path, "w") as f:
            json.dump({"method": args.method, "epochs": args.epochs,
                       "timestamp": datetime.now().isoformat(), "trials": results}, f, indent=2)

    total_time = time.time() - start_all
    print_report(results, args.method, total_time)

    valid = [r for r in results if r["val_acc"] is not None]
    valid.sort(key=lambda r: r["val_acc"], reverse=True)
    out_path = os.path.join(args.output, "search_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "method":       args.method,
            "epochs":       args.epochs,
            "n_trials":     len(results),
            "total_time_s": round(total_time, 1),
            "timestamp":    datetime.now().isoformat(),
            "best":         valid[0] if valid else None,
            "trials":       results,
        }, f, indent=2)

    print(f"\n  Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
