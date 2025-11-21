import yaml
from collections import Counter

primary_yaml = "../imagenet100_primary.yml"
seq_yaml = "../seq_imagenet_FG.yml"

# ---- Load primary YAML (class -> idx) ----
with open(primary_yaml, "r") as f:
    primary_mapping = yaml.safe_load(f)

primary_classes = set(primary_mapping.keys())

# ---- Load seq_imagenet_FG (group -> {class -> something}) ----
with open(seq_yaml, "r") as f:
    seq_mapping = yaml.safe_load(f)

# Collect all classes from all groups
seq_classes_list = []
for group_name, group_dict in seq_mapping.items():
    for cls in group_dict.keys():
        seq_classes_list.append(cls)

seq_classes = set(seq_classes_list)

# seq_classes = set(seq_mapping.keys())

# ---- Basic checks ----
missing_in_seq = primary_classes - seq_classes
extra_in_seq = seq_classes - primary_classes

intersection = primary_classes & seq_classes

# Duplicates (same class appears in more than one place)
counts = Counter(seq_classes_list)
duplicates = [cls for cls, c in counts.items() if c > 1]

print("=== CHECK seq_imagenet_FG vs imagenet100_primary ===")
print(f"Total classes in primary YAML: {len(primary_classes)}")
print(f"Total unique classes in seq   : {len(seq_classes)}")
# print(f"Total (with duplicates) in seq: {len(seq_classes_list)}\n")
print(f"Intersection count            : {len(intersection)}")
#check if all classes are the same
all_classes_same = (len(missing_in_seq) == 0) and (len(extra_in_seq) == 0)
print(f"All classes the same: {all_classes_same}\n")



# if missing_in_seq:
#     print(f"Classes present in primary YAML but NOT in seq_imagenet_FG: {len(missing_in_seq)}")
#     for cls in sorted(missing_in_seq):
#         print("  -", cls)
# else:
#     print("✔ No classes missing from seq_imagenet_FG.")

# print()

# if extra_in_seq:
#     print(f"Classes present in seq_imagenet_FG but NOT in primary YAML: {len(extra_in_seq)}")
#     for cls in sorted(extra_in_seq):
#         print("  -", cls)
# else:
#     print("✔ No extra classes in seq_imagenet_FG beyond primary YAML.")

# print()

# if duplicates:
#     print("Classes that appear MORE THAN ONCE in seq_imagenet_FG:")
#     for cls in sorted(duplicates):
#         print(f"  - {cls} (count = {counts[cls]})")
# else:
#     print("✔ No duplicates: each class appears exactly once in seq_imagenet_FG.")
