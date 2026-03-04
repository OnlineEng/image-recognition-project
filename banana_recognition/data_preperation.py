import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

rows = []

for row in range(1,21):
    filename = f"img_{row:03d}.jpg"
    rows.append({
        "images": f"data/images/{filename}",
        "labels": f"data/labels/train/img_{row:03d}.txt",
    })

data_df = pd.DataFrame(rows, columns=["images", "labels"])

print(data_df)

output_path = "/home/hayes/git_repos/image_recognition/banana_recognition/data/CSVs/data.csv"
data_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")