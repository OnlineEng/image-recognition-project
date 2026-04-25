import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor
from args import get_args
from model import build_model
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(checkpoint_path, device):
    args = get_args()
    model = build_model(args.backbone, num_classes=args.num_classes + 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, threshold=0.5):
    args = get_args()
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_resized = img.resize((args.image_size, args.image_size))
    tensor = to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    output = outputs[0]
    boxes = output['boxes'].cpu()
    scores = output['scores'].cpu()
    labels = output['labels'].cpu()

    # Filter by confidence threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Keep only the highest-scoring detection
    if len(scores) > 0:
        top_idx = scores.argmax().unsqueeze(0)
        boxes = boxes[top_idx]
        scores = scores[top_idx]
        labels = labels[top_idx]

    return img_resized, boxes, scores, labels

def draw_predictions(img, boxes, scores, labels, save_path=None):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                  linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 8), f"banana {score:.2f}",
                color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.show()
    plt.close()
    return img

def evaluate_folder(image_folder, checkpoint_path, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)

    output_folder = os.path.join(image_folder, 'predictions')
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith(image_extensions)
               and not f.startswith('pred_')]

    if not image_files:
        print("No images found in folder.")
        return

    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        print(f"Processing: {filename}")
        img, boxes, scores, labels = predict_image(model, image_path, device, threshold)

        if len(boxes) == 0:
            print(f"  No detections above threshold {threshold}")
        else:
            for i, (box, score) in enumerate(zip(boxes, scores)):
                print(f"  Detection {i+1}: score={score:.4f}, box={[round(x, 1) for x in box.tolist()]}")

        save_path = os.path.join(output_folder, f"pred_{filename}")
        draw_predictions(img, boxes, scores, labels, save_path=save_path)

if __name__ == "__main__":
    args = get_args()
    checkpoint = os.path.join(BASE_DIR, '..', 'sessions', 'best_model.pth')

    evaluate_folder(
        image_folder=os.path.join(BASE_DIR, 'data', 'test_images'),
        checkpoint_path=checkpoint,
        threshold=0.5
    )