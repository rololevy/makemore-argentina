"""
Training script for MakeMore Argentina.
Trains all model architectures on Argentine names and/or surnames datasets.

Usage:
    python train.py                              # Train all models on nombres (freq-weighted)
    python train.py --dataset apellidos           # Train all models on apellidos
    python train.py --model-type transformer      # Train only transformer
    python train.py --device cuda                 # Use GPU
    python train.py --max-steps 5000              # Custom step count
"""

import os
import sys
import time
import argparse
import json

import torch

from makemore_ar import (
    ModelConfig, create_model, create_datasets, InfiniteDataLoader,
    generate, MODEL_CLASSES, MODEL_DESCRIPTIONS
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default hyperparameters per model type
DEFAULT_CONFIGS = {
    'bigram': {'n_layer': 1, 'n_head': 1, 'n_embd': 64, 'n_embd2': 64, 'max_steps': 5000, 'lr': 5e-3},
    'mlp': {'n_layer': 1, 'n_head': 1, 'n_embd': 64, 'n_embd2': 128, 'max_steps': 10000, 'lr': 1e-3},
    'rnn': {'n_layer': 1, 'n_head': 1, 'n_embd': 64, 'n_embd2': 128, 'max_steps': 15000, 'lr': 5e-4},
    'gru': {'n_layer': 1, 'n_head': 1, 'n_embd': 64, 'n_embd2': 128, 'max_steps': 15000, 'lr': 5e-4},
    'bow': {'n_layer': 1, 'n_head': 1, 'n_embd': 64, 'n_embd2': 64, 'max_steps': 10000, 'lr': 5e-4},
    'transformer': {'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'max_steps': 15000, 'lr': 5e-4},
}

DATASET_FILES = {
    'nombres': os.path.join(SCRIPT_DIR, 'data', 'nombres_freq.txt'),
    'apellidos': os.path.join(SCRIPT_DIR, 'data', 'apellidos_freq.txt'),
}


def evaluate_model(model, dataset, device, batch_size=50, max_batches=10):
    """Evaluate model loss on a dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if i >= max_batches:
            break
    model.train()
    return torch.tensor(losses).mean().item()


def generate_samples(model, train_dataset, device, num=10, top_k=None):
    """Generate sample names and categorize them."""
    model.eval()
    X_init = torch.zeros(num, 1, dtype=torch.long).to(device)
    steps = train_dataset.get_output_length() - 1
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    train_samples, new_samples = [], []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word = train_dataset.decode(row)
        if train_dataset.contains(word):
            train_samples.append(word)
        else:
            new_samples.append(word)
    model.train()
    return train_samples, new_samples


def train_single_model(model_type, dataset_name, args):
    """Train a single model on a dataset."""
    input_file = DATASET_FILES[dataset_name]
    defaults = DEFAULT_CONFIGS[model_type]

    max_steps = args.max_steps if args.max_steps > 0 else defaults['max_steps']
    lr = args.learning_rate if args.learning_rate > 0 else defaults['lr']
    n_layer = defaults['n_layer']
    n_head = defaults['n_head']
    n_embd = defaults['n_embd']
    n_embd2 = defaults['n_embd2']
    device = args.device

    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} on {dataset_name}")
    print(f"{'='*70}")

    # Create datasets
    torch.manual_seed(args.seed)
    train_dataset, test_dataset = create_datasets(input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"vocab_size={vocab_size}, block_size={block_size}")

    # Create model
    config = ModelConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_embd2=n_embd2
    )
    model = create_model(model_type, config)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model #params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)

    # Data loader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=0)

    # Training loop
    best_loss = None
    output_dir = os.path.join(SCRIPT_DIR, 'models')
    os.makedirs(output_dir, exist_ok=True)

    model_filename = f"{dataset_name}_{model_type}.pt"
    model_path = os.path.join(output_dir, model_filename)
    training_log = []

    t_start = time.time()

    for step in range(max_steps):
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 100 == 0:
            print(f"  step {step}/{max_steps} | loss {loss.item():.4f} | {(t1-t0)*1000:.1f}ms/step")

        if step > 0 and step % 500 == 0:
            train_loss = evaluate_model(model, train_dataset, device)
            test_loss = evaluate_model(model, test_dataset, device)
            print(f"  step {step} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f}")

            training_log.append({
                'step': step,
                'train_loss': round(train_loss, 4),
                'test_loss': round(test_loss, 4),
            })

            if best_loss is None or test_loss < best_loss:
                print(f"  -> new best test loss {test_loss:.4f}, saving model")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                    'vocab_size': vocab_size,
                    'block_size': block_size,
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'n_embd': n_embd,
                    'n_embd2': n_embd2,
                    'chars': train_dataset.chars,
                    'max_word_length': train_dataset.max_word_length,
                    'dataset_file': os.path.basename(input_file),
                    'best_test_loss': test_loss,
                    'best_train_loss': train_loss,
                    'n_params': n_params,
                    'step': step,
                }
                torch.save(checkpoint, model_path)
                best_loss = test_loss

        if step > 0 and step % 2000 == 0:
            train_s, new_s = generate_samples(model, train_dataset, device, num=10)
            print(f"  Samples - existentes: {train_s[:5]} | nuevos: {new_s[:5]}")

    total_time = time.time() - t_start

    # Final eval
    final_train_loss = evaluate_model(model, train_dataset, device)
    final_test_loss = evaluate_model(model, test_dataset, device)

    # Generate final samples
    train_s, new_s = generate_samples(model, train_dataset, device, num=20)

    result = {
        'model_type': model_type,
        'dataset': dataset_name,
        'n_params': n_params,
        'best_test_loss': round(best_loss or final_test_loss, 4),
        'final_train_loss': round(final_train_loss, 4),
        'final_test_loss': round(final_test_loss, 4),
        'max_steps': max_steps,
        'training_time_sec': round(total_time, 1),
        'training_log': training_log,
        'sample_existing': train_s[:10],
        'sample_novel': new_s[:10],
    }

    print(f"\n  DONE: {model_type} on {dataset_name}")
    print(f"  Best test loss: {best_loss:.4f}")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  Params: {n_params:,}")
    print(f"  Saved to: {model_path}")
    print(f"  Existing samples: {train_s[:5]}")
    print(f"  Novel samples: {new_s[:5]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train MakeMore Argentina models")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['nombres', 'apellidos', 'all'],
                        help="Dataset to train on")
    parser.add_argument('--model-type', type=str, default='all',
                        choices=list(MODEL_CLASSES.keys()) + ['all'],
                        help="Model type to train")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device: cpu|cuda|mps")
    parser.add_argument('--max-steps', type=int, default=-1,
                        help="Override max training steps (-1 for default per model)")
    parser.add_argument('--learning-rate', type=float, default=-1,
                        help="Override learning rate (-1 for default per model)")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--seed', type=int, default=3407,
                        help="Random seed")
    args = parser.parse_args()

    # Determine which datasets and models to train
    datasets = ['nombres', 'apellidos'] if args.dataset == 'all' else [args.dataset]
    model_types = list(MODEL_CLASSES.keys()) if args.model_type == 'all' else [args.model_type]

    all_results = []

    for dataset_name in datasets:
        for model_type in model_types:
            result = train_single_model(model_type, dataset_name, args)
            all_results.append(result)

    # Save summary
    summary_path = os.path.join(SCRIPT_DIR, 'models', 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nTraining summary saved to {summary_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Dataset':<12} {'Params':>10} {'Best Test':>12} {'Time':>10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['model_type']:<15} {r['dataset']:<12} {r['n_params']:>10,} {r['best_test_loss']:>12.4f} {r['training_time_sec']:>8.1f}s")


if __name__ == '__main__':
    main()
