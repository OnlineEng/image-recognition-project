import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the absolute base path for your project
base_path = "/home/hayes/repos/image-recognition-project/banana_recognition/data"

# Set directories using the absolute base path
image_dir = os.path.join(base_path, "images")
label_dir = os.path.join(base_path, "labels/train")
output_dir = os.path.join(base_path, "CSVs")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

data_list = []

# Verify image directory exists before proceeding
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Could not find image directory at: {image_dir}")

# Dynamically read all images in the directory
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        # Only add to list if the label file actually exists
        if os.path.exists(label_path):
            data_list.append({
                "images": image_path,
                "labels": label_path
            })
        else:
            print(f"Warning: Missing label for {filename}")

# Create and save the master dataframe
data_df = pd.DataFrame(data_list)
data_output_path = os.path.join(output_dir, "data.csv")
data_df.to_csv(data_output_path, index=False)

# Split the dataset 80/20
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)

# Save the splits
train_output_path = os.path.join(output_dir, "train_df.csv")
val_output_path = os.path.join(output_dir, "val_df.csv")

train_df.to_csv(train_output_path, index=False)
val_df.to_csv(val_output_path, index=False)

# Output results for verification
print(f"Total images found: {len(data_df)}")
print(f"Saved {len(train_df)} training records to {train_output_path}")
print(f"Saved {len(val_df)} validation records to {val_output_path}")