import matplotlib.pyplot as plt

epochs = list(range(1, 31))
train_loss = [1.0923, 0.8819, 0.8010, 0.8020, 0.7577, 0.7319, 0.7366, 0.7776, 0.7715, 0.7111, 0.6445, 0.6530, 0.6449, 0.6865, 0.7244, 0.6832, 0.6779, 0.6924, 0.6544, 0.7441, 0.6364, 0.6317, 0.6644, 0.6002, 0.6607, 0.6186, 0.5004, 0.6736, 0.6428, 0.6373]
val_loss = [0.8922, 0.7055, 0.6511, 0.6400, 0.6563, 0.6029, 0.6262, 0.6819, 0.6473, 0.5500, 0.6226, 0.5748, 0.6338, 0.6851, 0.7327, 0.6526, 0.7579, 0.6726, 0.7767, 0.6856, 0.6514, 0.6935, 0.6227, 0.7293, 0.7824, 0.5176, 0.6709, 0.7018, 0.7795, 0.6536]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='royalblue', marker='o', markersize=4)
plt.plot(epochs, val_loss, label='Validation Loss', color='darkorange', marker='s', markersize=4)

# --- PINPOINT BEST VAL EPOCH ---
best_idx = val_loss.index(min(val_loss))
best_epoch = epochs[best_idx]
best_val = val_loss[best_idx]
# 1. Draw a prominent red dot on the exact coordinate
plt.scatter(best_epoch, best_val, color='red', s=100, zorder=5, edgecolors='black', label='Optimal Weights')
# 2. Draw a vertical dashed line to mark the early stopping point
plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
# 3. Add a clean text label without an arrow
# The 'x' and 'y' coordinates here dictate exactly where the bottom-left of the text starts.
plt.text(best_epoch + 0.5, best_val + 0.05, 
         f'Minimum Val Loss: {best_val:.4f}\n(Epoch {best_epoch})', 
         fontsize=10, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# -------------------------

plt.title('Training and Validation Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, 21, 2))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300)
print("Saved learning_curve.png")