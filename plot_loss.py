"""Generate train/val loss diff figure from log.txtt."""
import re
import matplotlib.pyplot as plt

train_iters, train_losses = [], []
val_iters, val_losses = [], []

with open("layerTrain/log.txtt") as f:
    for line in f:
        m = re.match(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", line)
        if m:
            it = int(m.group(1))
            tl, vl = float(m.group(2)), float(m.group(3))
            train_iters.append(it)
            train_losses.append(tl)
            val_iters.append(it)
            val_losses.append(vl)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: train + val loss
axes[0].plot(train_iters, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
axes[0].plot(val_iters, val_losses, label="Val Loss", color="#FF9800", linewidth=1.5)
axes[0].set_ylabel("Loss")
axes[0].set_title("Layer-Wise Training: Train vs Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bottom: diff (train - val)
diffs = [t - v for t, v in zip(train_losses, val_losses)]
axes[1].plot(train_iters, diffs, label="Train - Val", color="#4CAF50", linewidth=1.5)
axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss Diff")
axes[1].set_title("Train - Val Loss Gap (positive = train > val)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("layerTrain/loss_diff.png", dpi=150)
print("Saved: layerTrain/loss_diff.png")
plt.close()
