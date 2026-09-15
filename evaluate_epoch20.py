import os
import csv
from shapely.geometry import Polygon


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = r"D:\Ishani\DEAKIN UNI\T2-2026\SIT789_ROBOTICS\5.2HD_Datasets\ICDAR2015\ICDAR2015_CUSTOM"

GT_DIR = os.path.join(DATASET_ROOT, "test_gt")

PRED_DIR = r"D:\Ishani\DEAKIN UNI\T2-2026\SIT789_ROBOTICS\5.2HD\EAST\test_results_epoch20"

OUTPUT_CSV = os.path.join(PRED_DIR, "evaluation_epoch20.csv")

IOU_THRESHOLD = 0.5


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(file_path):

    valid_polygons = []
    ignored_polygons = []

    with open(file_path, "r", encoding="utf-8-sig") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 9:
                continue

            try:

                coords = list(map(float, parts[:8]))

                points = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7])
                ]

                polygon = Polygon(points)

                if not polygon.is_valid:
                    polygon = polygon.buffer(0)

                if polygon.area <= 0:
                    continue

                text = ",".join(parts[8:]).strip()

                if "###" in text:
                    ignored_polygons.append(polygon)
                else:
                    valid_polygons.append(polygon)

            except ValueError:
                continue

    return valid_polygons, ignored_polygons


# ============================================================
# READ PREDICTIONS
# ============================================================

def read_predictions(file_path):

    polygons = []

    if not os.path.exists(file_path):
        return polygons

    with open(file_path, "r", encoding="utf-8-sig") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 8:
                continue

            try:

                coords = list(map(float, parts[:8]))

                points = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7])
                ]

                polygon = Polygon(points)

                if not polygon.is_valid:
                    polygon = polygon.buffer(0)

                if polygon.area <= 0:
                    continue

                polygons.append(polygon)

            except ValueError:
                continue

    return polygons


# ============================================================
# CALCULATE IoU
# ============================================================

def calculate_iou(poly1, poly2):

    try:

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union <= 0:
            return 0.0

        return intersection / union

    except Exception:
        return 0.0


# ============================================================
# CHECK OVERLAP WITH IGNORED REGION
# ============================================================

def overlaps_ignored(prediction, ignored_polygons):

    if not ignored_polygons:
        return False

    for ignored in ignored_polygons:

        try:

            intersection = prediction.intersection(ignored).area

            if prediction.area > 0:

                overlap_ratio = intersection / prediction.area

                if overlap_ratio >= 0.5:
                    return True

        except Exception:
            continue

    return False


# ============================================================
# EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(gt_file, pred_file):

    gt_polygons, ignored_polygons = read_ground_truth(gt_file)

    predictions = read_predictions(pred_file)

    # Remove predictions that mostly fall inside ### regions
    filtered_predictions = []

    ignored_prediction_count = 0

    for prediction in predictions:

        if overlaps_ignored(prediction, ignored_polygons):

            ignored_prediction_count += 1

        else:

            filtered_predictions.append(prediction)

    predictions = filtered_predictions

    matched_gt = set()
    matched_pred = set()

    matches = []

    # --------------------------------------------------------
    # Create all possible matches
    # --------------------------------------------------------

    possible_matches = []

    for pred_index, prediction in enumerate(predictions):

        for gt_index, ground_truth in enumerate(gt_polygons):

            iou = calculate_iou(prediction, ground_truth)

            if iou >= IOU_THRESHOLD:

                possible_matches.append(
                    (iou, pred_index, gt_index)
                )

    # --------------------------------------------------------
    # Highest IoU matches first
    # --------------------------------------------------------

    possible_matches.sort(reverse=True)

    for iou, pred_index, gt_index in possible_matches:

        if pred_index in matched_pred:
            continue

        if gt_index in matched_gt:
            continue

        matched_pred.add(pred_index)
        matched_gt.add(gt_index)

        matches.append(iou)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    tp = len(matched_gt)

    fp = len(predictions) - tp

    fn = len(gt_polygons) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    average_iou = (
        sum(matches) / len(matches)
        if matches
        else 0.0
    )

    return {
        "gt": len(gt_polygons),
        "predictions": len(predictions),
        "ignored_predictions": ignored_prediction_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_iou": average_iou
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EAST MODEL EVALUATION - EPOCH 20")
    print("=" * 70)

    print()
    print("Ground truth directory:")
    print(GT_DIR)

    print()
    print("Prediction directory:")
    print(PRED_DIR)

    print()
    print("IoU threshold:", IOU_THRESHOLD)

    if not os.path.exists(GT_DIR):

        print()
        print("ERROR: Ground truth directory does not exist.")
        return

    if not os.path.exists(PRED_DIR):

        print()
        print("ERROR: Prediction directory does not exist.")
        return

    gt_files = sorted(
        [
            f for f in os.listdir(GT_DIR)
            if f.lower().endswith(".txt")
        ]
    )

    print()
    print("Ground truth files found:", len(gt_files))

    if len(gt_files) == 0:

        print("ERROR: No ground truth files found.")
        return

    total_gt = 0
    total_predictions = 0
    total_ignored = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    image_results = []

    # ========================================================
    # IMAGE-BY-IMAGE EVALUATION
    # ========================================================

    for gt_filename in gt_files:

        gt_path = os.path.join(
            GT_DIR,
            gt_filename
        )

        # gt_img_121.txt -> res_img_121.txt
        image_id = gt_filename

        if image_id.startswith("gt_"):
            image_id = image_id[3:]

        pred_filename = "res_" + image_id

        pred_path = os.path.join(
            PRED_DIR,
            pred_filename
        )

        result = evaluate_image(
            gt_path,
            pred_path
        )

        total_gt += result["gt"]
        total_predictions += result["predictions"]
        total_ignored += result["ignored_predictions"]

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        image_results.append(
            (
                gt_filename,
                result
            )
        )

        print(
            "{:<20} GT={:<3} Pred={:<3} "
            "TP={:<3} FP={:<3} FN={:<3} "
            "F1={:.3f}".format(
                gt_filename,
                result["gt"],
                result["predictions"],
                result["tp"],
                result["fp"],
                result["fn"],
                result["f1"]
            )
        )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print()
    print("Total ground-truth boxes :", total_gt)
    print("Total predictions        :", total_predictions)
    print("Ignored predictions      :", total_ignored)

    print()
    print("True Positives           :", total_tp)
    print("False Positives          :", total_fp)
    print("False Negatives          :", total_fn)

    print()
    print("Precision                : {:.4f}".format(precision))
    print("Recall                   : {:.4f}".format(recall))
    print("F1-score                 : {:.4f}".format(f1))

    print()
    print("Precision (%)            : {:.2f}".format(
        precision * 100
    ))

    print("Recall (%)               : {:.2f}".format(
        recall * 100
    ))

    print("F1-score (%)             : {:.2f}".format(
        f1 * 100
    ))

    print()
    print("=" * 70)

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "image",
            "ground_truth",
            "predictions",
            "ignored_predictions",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "f1",
            "average_iou"
        ])

        for filename, result in image_results:

            writer.writerow([
                filename,
                result["gt"],
                result["predictions"],
                result["ignored_predictions"],
                result["tp"],
                result["fp"],
                result["fn"],
                result["precision"],
                result["recall"],
                result["f1"],
                result["average_iou"]
            ])

        writer.writerow([])

        writer.writerow([
            "OVERALL",
            total_gt,
            total_predictions,
            total_ignored,
            total_tp,
            total_fp,
            total_fn,
            precision,
            recall,
            f1,
            ""
        ])

    print()
    print("CSV saved to:")
    print(OUTPUT_CSV)
    print()


if __name__ == "__main__":
    main()