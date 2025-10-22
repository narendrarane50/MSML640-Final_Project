# main.py
import argparse
from training.ablation_runner import run_ablation

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--dataset", type=str, default="rafdb", choices=["rafdb","fer2013","affectnet"])
    args = ap.parse_args()

    results = run_ablation(cfg_path=args.config, dataset=args.dataset)
    print("\n=== Ablation Summary ===")
    for k, v in results.items():
        print(k, {m: (round(x,4) if m!="confusion_matrix" else "\n"+str(x)) for m, x in v.items()})
