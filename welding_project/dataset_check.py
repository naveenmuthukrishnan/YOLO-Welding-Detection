#Check total number of images in each folder(train, test, validation)
#Colab automatically contains our data inside content folder---so running in colab needs /content/
import os 
train_img_counts = len(os.listdir(path = "datasets/train/images"))

test_img_counts = len(os.listdir(path = "datasets/test/images"))

val_img_counts = len(os.listdir(path = "datasets/valid/images"))


#To check label counts

from pathlib import Path
from collections import Counter

train_counts = Counter(
    int(line.split()[0]) for p in Path("datasets/train/labels").glob("*.txt") for line in p.read_text().splitlines() if line.strip()
)
good_train_labels = train_counts[1]
defective_train_labels = train_counts[0] + train_counts[2]

test_counts = Counter(
    int(line.split()[0]) for p in Path("datasets/test/labels").glob("*.txt") for line in p.read_text().splitlines() if line.strip()
)
good_test_labels = test_counts[1]
defective_test_labels = test_counts[0] + test_counts[2]

val_counts = Counter(
    int(line.split()[0]) for p in Path("datasets/valid/labels").glob("*.txt") for line in p.read_text().splitlines() if line.strip()
)
good_valid_labels = val_counts[1]
defective_valid_labels = val_counts[0] + val_counts[2]

#now counts

print("Total images for training:", train_img_counts)
print("Total images for testing:", test_img_counts)
print("Total images for validation:", val_img_counts)
print(f"Total good train labels: {good_train_labels} || Total defect train labels: {defective_train_labels}")
print(f"Total good test labels: {good_test_labels} || Total defect test labels: {defective_test_labels}")
print(f"Total good val labels: {good_valid_labels} || Total defect val labels: {defective_valid_labels}")