import json
import os
import shutil

images_dir = "data/mpii/images"
validation_dir = os.path.join(images_dir, "val")
json_file = "data/mpii/annotations/mpii_val.json"

os.makedirs(validation_dir, exist_ok=True)

with open(json_file, "r") as f:
    data = json.load(f)

for item in data:
    image_name = item.get("image")
    if not image_name:
        continue

    source_path = os.path.join(images_dir, image_name)
    dest_path = os.path.join(validation_dir, image_name)

    if os.path.exists(source_path):
        shutil.move(source_path, dest_path)
        print(f"Spostata: {image_name}")
    else:
        print(f"File non trovato: {image_name}")

print("Operazione completata.")
