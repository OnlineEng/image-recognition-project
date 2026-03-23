import matplotlib.pyplot as plt
epochs = list(range(1, 26))
train_loss = [1.7371, 1.3616, 1.1513, 1.0606, 1.0590, 1.0407, 1.0562, 1.0837, 1.0815, 1.0405, 0.9798, 0.9243, 0.8604, 0.8445, 0.8171, 0.7819, 0.7056, 0.6510, 0.6157, 0.5823, 0.5597, 0.5232, 0.4898, 0.4651, 0.4384]
val_loss = [1.2612, 1.1825, 1.1055, 1.0288, 1.0859, 1.0157, 1.0077, 1.0051, 1.0435, 1.0114, 0.9855, 0.9591, 1.0132, 1.0470, 1.0443, 1.0447, 1.0912, 1.1003, 1.1075, 1.0941, 1.1607, 1.1214, 1.1306, 1.1923, 1.1760]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='royalblue', marker='o', markersize=4)
plt.plot(epochs, val_loss, label='Validation Loss', color='darkorange', marker='s', markersize=4)

plt.title('Training and Validation Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300)
print("Saved learning_curve.png")