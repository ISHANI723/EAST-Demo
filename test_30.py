import os
import torch
from PIL import Image
from model import EAST
from detect import detect, plot_boxes


TEST_DIR = r"D:\Ishani\DEAKIN UNI\T2-2026\SIT789_ROBOTICS\5.2HD_Datasets\ICDAR2015\ICDAR2015_CUSTOM\test_img"

OUTPUT_DIR = "./test_results_epoch20"
os.makedirs(OUTPUT_DIR, exist_ok=True)


device = torch.device("cpu")

print("Loading trained EAST model...")

model = EAST(False)

checkpoint = torch.load(
    "./pths/model_epoch_20.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully.")
print()


image_files = sorted(
    [
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(".jpg")
    ]
)

print("Test images:", len(image_files))
print("=" * 60)


total_detections = 0

for i, filename in enumerate(image_files, start=1):

    image_path = os.path.join(TEST_DIR, filename)

    img = Image.open(image_path).convert("RGB")

    boxes = detect(img, model, device)

    num_boxes = 0 if boxes is None else len(boxes)

    total_detections += num_boxes

    output_path = os.path.join(
        OUTPUT_DIR,
        "result_" + filename
    )

    plot_boxes(img, boxes).save(output_path)

    print(
        f"[{i:02d}/{len(image_files)}] "
        f"{filename} -> {num_boxes} detections"
    )


print("=" * 60)
print("Testing completed.")
print("Total detections:", total_detections)
print("Average detections per image:",
      total_detections / len(image_files))

print("Results saved to:")
print(os.path.abspath(OUTPUT_DIR))