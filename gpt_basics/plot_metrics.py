import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_history(filename:str, split: str, save_path: str = None):
    """
    Plots train and validation loss curves for multiple experiments.

    Args:
        split (str): The split to plot ("train" or "val").
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    plt.figure(figsize=(10, 4))

    import json 
    with open(filename) as f:
        history = json.load(f)

    for exp_name, hist in history.items():
        losses = hist.get(split, [])    


        if losses:
            plt.plot(losses, label=f"{exp_name}")

    plt.xlabel("Evaluation Interval")
    plt.ylabel("Loss")
    plt.title(f"{split.capitalize()} Loss Across Experiments")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_metrics(metrics_json_path, output_path=None, show=True):
    """Load metrics JSON and plot bar charts comparing experiments.

    The metrics JSON format is expected to be a mapping from experiment tag
    to a list of dicts with keys: 'step', 'ms_per_step', 'tok_per_sec'.
    """
    if not os.path.isfile(metrics_json_path):
        raise FileNotFoundError(metrics_json_path)

    with open(metrics_json_path, 'r') as f:
        data = json.load(f)

    tags = []
    avg_ms = []
    avg_tok = []

    for tag, entries in data.items():
        if not entries:
            continue
        tags.append(tag)
        ms_vals = [e.get('ms_per_step', 0.0) for e in entries]
        tok_vals = [e.get('tok_per_sec', 0.0) for e in entries]
        avg_ms.append(float(np.mean(ms_vals)))
        avg_tok.append(float(np.mean(tok_vals)))

    if not tags:
        raise ValueError('No metrics found in file')

    x = np.arange(len(tags))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(6, len(tags)*1.5), 4))

    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, avg_ms, width, label='avg ms/step', color='#1f77b4')
    bars2 = ax2.bar(x + width/2, avg_tok, width, label='avg tok/s', color='#ff7f0e')

    ax1.set_xlabel('Experiment')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tags, rotation=45, ha='right')
    ax1.set_ylabel('ms / step')
    ax2.set_ylabel('tokens / sec')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Experiment Performance: avg ms/step and tok/s')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('metrics_json')
    p.add_argument('--out', '-o', help='Output image path (optional)')
    args = p.parse_args()

    plot_metrics(args.metrics_json, output_path=args.out, show=True)


def plot_metrics_history(metrics_json_path='metrics.json', metric='ms_per_step', save_path=None):
    """Plot a time series of a single metric for each experiment, like `plot_history`.

    Args:
        metrics_json_path: path to the metrics JSON file.
        metric: one of 'ms_per_step' or 'tok_per_sec'.
        save_path: optional output path to save the figure.
    """
    import json
    import matplotlib.pyplot as plt

    if not os.path.isfile(metrics_json_path):
        raise FileNotFoundError(metrics_json_path)

    with open(metrics_json_path, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(10, 4))

    for exp_name, entries in data.items():
        vals = [e.get(metric, None) for e in entries]
        if vals:
            plt.plot(vals, label=exp_name)

    plt.xlabel('Evaluation Interval')
    if metric == 'ms_per_step':
        plt.ylabel('ms / step')
        plt.title('ms/step Across Experiments')
    else:
        plt.ylabel('tokens / sec')
        plt.title('tokens/sec Across Experiments')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()
