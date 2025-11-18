import numpy as np
import yaml

yaml_file = "../imagenet100_aux.yml"
npy_file = "imagenet_100_subset_classes_list.npy"

# 1. Load YAML
with open(yaml_file, "r") as f:
    mapping = yaml.safe_load(f)

yaml_classes = set(mapping.keys())

# 2. Load the .npy file
subset_classes = np.load(npy_file)

# ensure everything is a proper string
subset_classes = set(map(str, subset_classes))

# 3. Compute intersections and differences
intersection = yaml_classes & subset_classes
missing_from_subset = yaml_classes - subset_classes
extras_in_subset = subset_classes - yaml_classes

print("=== RESULTS ===")
print(f"YAML classes:      {len(yaml_classes)}")
print(f"NPY subset classes:{len(subset_classes)}")
print(f"Intersection:      {len(intersection)}\n")

if intersection:
    print("Common classes:")
    for c in sorted(intersection):
        print("  -", c)
else:
    print("No common classes.")

if missing_from_subset:
    print("Classes in YAML but not in the .npy subset:")
    for c in sorted(missing_from_subset):
        print("  -", c)
else:
    print("No YAML classes missing from .npy.")

print()

if extras_in_subset:
    print("Classes in .npy subset but not in YAML:")
    for c in sorted(extras_in_subset):
        print("  -", c)
else:
    print("No extra classes in .npy beyond YAML.")


""" 
=== RESULTS === imagenet100_primary.yml AND imagenet_100_subset_classes_list.npy
YAML classes:      100
NPY subset classes:100
Intersection:      0


=== RESULTS === imagenet100_aux.yml AND imagenet_100_subset_classes_list.npy
YAML classes:      100
NPY subset classes:100
Intersection:      17

Common classes:
  - n02974003
  - n03967562
  - n03976657
  - n04004767
  - n04019541
  - n04033995
  - n04074963
  - n04099969
  - n04118538
  - n04125021
  - n04141327
  - n04162706
  - n04179913
  - n04192698
  - n04259630
  - n04330267
  - n09428293
    """