import os
import yaml

# paths
# yaml_file = "../seq_miniimg.yml"
yaml_file = "../imagenet100_primary.yml"
train_dir = "/mnt/storage_6TB/share/data/ImageNetFG/annotations/train"

# load YAML
with open(yaml_file, "r") as f:
    mapping = yaml.safe_load(f)

yaml_classes = set(mapping.keys())

# list directories inside train/
train_classes = set(
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
)

# compute differences
missing_dirs = yaml_classes - train_classes
extra_dirs = train_classes - yaml_classes
intersection = yaml_classes & train_classes

print("=== CHECK RESULTS ===")
print(f"Total YAML classes : {len(yaml_classes)}")
print(f"Total train dirs   : {len(train_classes)}\n")
print(f"Intersection count : {len(intersection)}\n")

# if missing_dirs:
#     print(f"Missing directories (present in YAML but not in train/): {len(missing_dirs)}")
#     for d in sorted(missing_dirs):
#         print("  -", d)
# else:
#     print("No missing directories.")

# print()

# if extra_dirs:
#     print(f"Extra directories (present in train/ but not YAML): {len(extra_dirs)}")
#     for d in sorted(extra_dirs):
#         print("  -", d)
# else:
#     print("No extra directories.")


"""
=== CHECK RESULTS === imagenet100_primary.yml AND ImageNetFG/images/train/ ===
Total YAML classes : 100
Total train dirs   : 100

No missing directories.

No extra directories.
"""