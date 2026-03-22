"""Build unified training summary from all model checkpoint files."""
import json
import torch
import os

models_dir = 'models'
summary = []
for fname in sorted(os.listdir(models_dir)):
    if not fname.endswith('.pt'):
        continue
    path = os.path.join(models_dir, fname)
    cp = torch.load(path, map_location='cpu', weights_only=False)
    entry = {
        'model_type': cp['model_type'],
        'dataset': cp['dataset_name'],
        'n_params': cp['n_params'],
        'best_test_loss': round(cp['best_test_loss'], 4),
        'final_train_loss': round(cp.get('best_train_loss', 0), 4),
        'final_test_loss': round(cp['best_test_loss'], 4),
        'max_steps': cp.get('step', 0),
        'training_time_sec': 0,
        'training_log': [],
        'sample_existing': [],
        'sample_novel': [],
    }
    summary.append(entry)
    print(f"{fname}: {cp['model_type']} on {cp['dataset_name']}, "
          f"loss={cp['best_test_loss']:.4f}, step={cp.get('step', 0)}")

with open(os.path.join(models_dir, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Wrote summary with {len(summary)} entries")
