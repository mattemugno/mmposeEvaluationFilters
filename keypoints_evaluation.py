import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

COCO_METRICS = {
    0: 'AP',
    1: 'AP. 5',
    2: 'AP. 75',
    3: 'AP(M)',
    4: 'AP(L)',
    5: 'AR',
    6: 'AR. 5',
    7: 'AR. 75',
    8: 'AR(M)',
    9: 'AR(L)',
}

skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def load_data(gt_path, pred_path):
    """
    Load gt Json file and predictions Json file.
    :param gt_path: Path al file JSON del ground truth.
    :param pred_path: Path al file JSON delle predizioni.
    :return: COCO objects for gt and predictions.
    """
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    return coco_gt, coco_dt


def evaluate(coco_gt, coco_dt, img_ids=None, iou_type='keypoints'):
    """
    Evaluation using COCOEval API.
    :param coco_gt: COCO ground truth object.
    :param coco_dt: COCO prediction object.
    :param img_ids: List of id images to evaluate (optional).
    :param iou_type: Evaluation type ('bbox', 'segm', 'keypoints').
    :return: COCOeval object with results for the given images.
    """

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if img_ids:
        coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


if __name__ == "__main__":

    data_root = "data/coco/annotations"
    results_root = "tools/json_results"
    model = "hrnet-384x288"
    filter = "pixelate"

    for pred_file in os.listdir(f'tools/json_results/{model}/{filter}/format_only'):

        filter_intensity = pred_file.split(".")[0]

        gt_path = os.path.join(data_root, "person_keypoints_val2017.json")
        pred_path = os.path.join(results_root, model, filter, 'format_only', pred_file)

        coco_gt, coco_dt = load_data(gt_path, pred_path)

        coco_eval = evaluate(coco_gt, coco_dt, iou_type='keypoints')

        keypoint_distances = {}

        for pred_on_image in coco_eval.evalImgs:
            if pred_on_image is None:
                continue
            keypoint_errors = pred_on_image['keypoint_errors']
            for gt_id, predictions in keypoint_errors.items():
                for pred_id, metrics in predictions.items():
                    for kp_idx, distance in enumerate(metrics['distance']):
                        if kp_idx not in keypoint_distances:
                            keypoint_distances[kp_idx] = []
                        keypoint_distances[kp_idx].append(distance)

        mean_errors = {}
        for kp_idx, distances in keypoint_distances.items():
            if distances:
                mean_errors[kp_idx] = sum(distances) / len(distances)
            else:
                mean_errors[kp_idx] = None

        json_dict = {}

        for i, metric in enumerate(coco_eval.stats):
            json_dict[COCO_METRICS[i]] = metric

        for kp_idx, mean_error in mean_errors.items():
            print(f"Mean error for {COCO_KEYPOINT_INDEXES[kp_idx]}: {mean_error}")
            json_dict[f"{COCO_KEYPOINT_INDEXES[kp_idx]}"] = mean_error

        with open(f"{results_root}/{model}/{filter}/results/results_{model}_{filter_intensity}.json", "w") as f:
            json.dump(json_dict, f, indent=2)