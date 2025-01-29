# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
from argparse import ArgumentParser
from os.path import exists

import cv2
import numpy as np
from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, bbox_xywh2xyxy


def pixelate(img, pixel_size=5):
    h, w = img.shape[:2]

    # Calculate the size of the reduced image
    reduced_w = max(1, w // pixel_size)
    reduced_h = max(1, h // pixel_size)

    img_small = cv2.resize(img, (reduced_w, reduced_h), interpolation=cv2.INTER_LINEAR)
    img_pixelated = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)

    return img_pixelated


def _load_detection_results(bbox_file, img_id):
    """Load data from detection results with dummy keypoint annotations."""

    assert exists(bbox_file), (f'Bbox file `{bbox_file}` does not exist')
    # load detection results
    with open(bbox_file, 'r') as f:
        det_results = json.load(f)
    det_results = [entry for entry in det_results if entry["image_id"] == img_id]

    data_list = []
    for det in det_results:
        # remove non-human instances
        if det['category_id'] != 1:
            continue

        bbox_xywh = np.array(det['bbox'][:4], dtype=np.float32).reshape(1, 4)
        bbox = bbox_xywh2xyxy(bbox_xywh)
        bbox_score = np.array(det['score'], dtype=np.float32).reshape(1)

        if bbox_score > 0.9:
            data_list.append(bbox[0])

    return data_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    bboxxes_path = "../data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"

    kernel_sizes = [5, 17, 29, 41, 53, 65, 77, 89, 101]
    pixel_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    for intensity in pixel_sizes:

        model = init_model(
            #args.config + f'_{intensity}x{intensity}.py',
            args.config + f'_{intensity}.py',
            args.checkpoint,
            device=args.device,
            cfg_options=cfg_options)

        # init visualizer
        model.cfg.visualizer.radius = args.radius
        model.cfg.visualizer.alpha = args.alpha
        model.cfg.visualizer.line_width = args.thickness

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.set_dataset_meta(
            model.dataset_meta, skeleton_style=args.skeleton_style)

        id = int(args.img.split('/')[-1].split('.')[0])
        bboxes = _load_detection_results(bboxxes_path, id)


        # inference a single image
        batch_results = inference_topdown(model, args.img, bboxes=bboxes)
        results = merge_data_samples(batch_results)

        # show the results
        img = imread(args.img, channel_order='rgb')
        visualizer.add_datasample(
            'result',
            #cv2.GaussianBlur(img, (intensity, intensity), 0),
            pixelate(img, pixel_size=intensity),
            data_sample=results,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            draw_heatmap=args.draw_heatmap,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            out_file=f'{intensity}.jpg')

        if args.out_file is not None:
            print_log(
                f'the output image has been saved at {args.out_file}',
                logger='current',
                level=logging.INFO)


if __name__ == '__main__':
    main()
