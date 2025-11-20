import os
import yaml

# paths
yaml_file = "../seq_miniimg.yml"
train_dir = "/mnt/storage_6TB/share/data/ImageNetFG/images/train"

# load YAML
with open(yaml_file, "r") as f:
    mapping = yaml.safe_load(f)

yaml_classes = set(mapping.keys())

# list directories inside train/
train_classes = set(
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
)

# compute intersections
intersection = yaml_classes & train_classes

print("=== CHECK RESULTS ===")
print(f"Total YAML classes : {len(yaml_classes)}")
print(f"Total train dirs   : {len(train_classes)}")
print(f"Intersection count : {len(intersection)}\n")

if intersection:
    print("Common classes:")
    for c in sorted(intersection):
        print("  -", c)
else:
    print("No common classes.")