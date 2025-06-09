import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from anomalib import TaskType
from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib.metrics import MinMax
from anomalib.utils.normalization.min_max import normalize


def process_and_save_image(image_path, left_dir, middle_dir, right_dir, resize_dims, crop_dimensions):
    with Image.open(image_path) as img:
        img = img.resize(resize_dims, Image.Resampling.LANCZOS)
        img_width, img_height = img.size
        crop_width, crop_height = crop_dimensions
        left = (img_width - crop_width) // 2
        upper = (img_height - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height
        cropped_img = img.crop((left, upper, right, lower))

        left_patch_width = int(crop_width * 0.3)
        middle_patch_width = int(crop_width * 0.4)
        right_patch_width = crop_width - left_patch_width - middle_patch_width

        left_box = (0, 0, left_patch_width, crop_height)
        middle_box = (left_patch_width, 0, left_patch_width + middle_patch_width, crop_height)
        right_box = (left_patch_width + middle_patch_width, 0, crop_width, crop_height)

        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)

        cropped_img.crop(left_box).save(os.path.join(left_dir, f"{name}_left{ext}"))
        cropped_img.crop(middle_box).save(os.path.join(middle_dir, f"{name}_middle{ext}"))
        cropped_img.crop(right_box).save(os.path.join(right_dir, f"{name}_right{ext}"))


def process_images_in_directory(source_dir, output_base_dir, resize_dimensions, crop_dimensions):
    left_dir = os.path.join(output_base_dir, 'left_patches')
    middle_dir = os.path.join(output_base_dir, 'middle_patches')
    right_dir = os.path.join(output_base_dir, 'right_patches')
    print(left_dir)

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(middle_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            process_and_save_image(
                os.path.join(source_dir, filename),
                left_dir, middle_dir, right_dir,
                resize_dimensions, crop_dimensions
            )

import shutil

def clear_split_image_directory(split_base_dir):
    """
    Deletes all files in the left_patches, middle_patches, and right_patches subdirectories.
    """
    subdirs = ['left_patches', 'middle_patches', 'right_patches']
    for subdir in subdirs:
        dir_path = os.path.join(split_base_dir, subdir)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Cleared directory: {dir_path}")
        else:
            print(f"Directory does not exist and cannot be cleared: {dir_path}")

def predic(dataset_dir, base_dir, model_pth):
    dataset = PredictDataset(path=dataset_dir + base_dir)
    dataloader = DataLoader(dataset=dataset)
    model = Patchcore.load_from_checkpoint(
        os.path.join(model_pth + base_dir, "Patchcore/RESIZE_NORMALIZED/v1/weights/lightning/model.ckpt"))
    engine = Engine(
        normalization=NormalizationMethod.NONE,
        task=TaskType.SEGMENTATION,
        image_metrics=["AUROC"],
        pixel_metrics=["AUROC", "PRO"]
    )
    preds_ = engine.predict(model=model, dataloaders=dataloader)
    img_thresh = model.image_threshold.value.item()
    pxl_thresh = model.pixel_threshold.value.item()
    return preds_, (img_thresh, pxl_thresh)


def run_pipeline(model_path, image_input_path):
    batch_sz = 4
    crop_dimensions = (1236, 300)
    resize_dimensions = (1280, 720)
    split_patch_dir = image_input_path+"split_patches"

    # Step 1: Crop + Split
    process_images_in_directory(image_input_path, split_patch_dir, resize_dimensions, crop_dimensions)

    # Step 2: Predict per split
    dirs = ['/left_patches', '/middle_patches', '/right_patches']
    preds, thres = [], []

    for d in dirs:
        prds, thrs = predic(split_patch_dir, d, model_path)
        preds.append(prds)
        thres.append(thrs)

    # Step 3: Normalize predictions
    minmax_pred_score = MinMax()
    minmax_anomaly_score = MinMax()

    for left, middle, right in zip(preds[0], preds[1], preds[2]):
        pred_scores_stacked = torch.stack([left['pred_scores'], middle['pred_scores'], right['pred_scores']])
        anomaly_scores_stacked = torch.cat((left['anomaly_maps'], middle['anomaly_maps'], right['anomaly_maps']), dim=3)
        minmax_pred_score.update(pred_scores_stacked)
        minmax_anomaly_score.update(anomaly_scores_stacked)

    pred_score_min, pred_score_max = minmax_pred_score.compute()
    anomaly_score_min, anomaly_score_max = minmax_anomaly_score.compute()

    preds_com = []
    for pred, thrs in zip(preds, thres):
        tmp = []
        for batches in pred:
            new_batch = {
                'image': batches['image'],
                'image_path': batches['image_path'],
                'pred_threshold': thrs[0],
                'anomaly_threshold': thrs[1],
                'pred_nonormalized': batches['pred_scores'],
                'anomaly_nonormalized_max': batches['anomaly_maps'].max(),
                'pred_labels': batches['pred_labels'],
                'pred_masks': batches['pred_masks'],
                'pred_scores': normalize(batches['pred_scores'], thrs[0], pred_score_min, pred_score_max),
                'anomaly_maps': normalize(batches['anomaly_maps'], thrs[1], anomaly_score_min, anomaly_score_max),
                'segments': [len(t.tolist()) for t in batches["box_labels"]]
            }
            tmp.append(new_batch)
        preds_com.append(tmp)

    preds_combined = []
    for pred, thrs in zip(preds_com, thres):
        tmp = []
        for batches in pred:
            tmp.append({
                'image': batches['image'],
                'image_path': batches['image_path'],
                'pred_threshold': thrs[0],
                'anomaly_threshold': thrs[1],
                'pred_nonormalized': batches['pred_nonormalized'],
                'anomaly_nonormalized_max': batches['anomaly_nonormalized_max'],
                'pred_labels': batches['pred_labels'],
                'pred_masks': batches['pred_masks'],
                'pred_scores': batches['pred_scores'],
                'anomaly_maps': batches['anomaly_maps'],
                'segments': batches["segments"]
            })
        preds_combined.append(tmp)

    test_dat = []
    for left, middle, right in zip(preds_combined[0], preds_combined[1], preds_combined[2]):
        data = {
            'image_path': [[li, mi, ri] for li, mi, ri in zip(left['image_path'], middle['image_path'], right['image_path'])],
            'pred_scores': torch.max(torch.stack([left['pred_scores'], middle['pred_scores'], right['pred_scores']]), dim=0).values,
            'pred_labels': torch.max(torch.stack([left['pred_labels'], middle['pred_labels'], right['pred_labels']]), dim=0).values,
            'pred_masks': torch.cat((left['pred_masks'], middle['pred_masks'], right['pred_masks']), dim=3),
            'anomaly_maps': torch.cat((left['anomaly_maps'], middle['anomaly_maps'], right['anomaly_maps']), dim=3),
            'segments': [left['segments'][i] + middle['segments'][i] + right['segments'][i] for i in range(len(left['segments']))],
            'pred_threshold': [left['pred_threshold'], middle['pred_threshold'], right['pred_threshold']],
            'anomaly_threshold': [left['anomaly_threshold'], middle['anomaly_threshold'], right['anomaly_threshold']],
            'pred_nonormalized': torch.max(torch.stack([left['pred_nonormalized'], middle['pred_nonormalized'], right['pred_nonormalized']]), dim=0).values.tolist(),
            'anomaly_nonormalized': torch.max(torch.stack([left['anomaly_nonormalized_max'], middle['anomaly_nonormalized_max'], right['anomaly_nonormalized_max']]), dim=0).values.tolist()
        }
        test_dat.append(data)

    return test_dat,split_patch_dir


def visualiser(predictions, sample_idx, output_dir):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    cv2_images = [cv2.imread(p) for p in predictions["image_path"][sample_idx]]
    cv2_stacked_image = cv2.hconcat(cv2_images)
    anomaly_map = predictions["anomaly_maps"][0].permute(1, 2, 0).cpu().numpy()
    heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=cv2_stacked_image, normalize=False)
    pred_mask = predictions["pred_masks"][0].permute(1, 2, 0).cpu().numpy()
    pred_label = predictions["pred_labels"][0]
    total_preds = predictions["segments"][0]
    patch_thresholds = predictions["pred_threshold"]
    image_score = predictions["pred_nonormalized"][0]
    pixel_score = predictions["anomaly_nonormalized"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Prediction: {'Anomalous' if pred_label else 'Normal'}\n"
                 f"Image patches thresholds left to right {patch_thresholds}\n"
                 f"Image Score {image_score}, Pixel score {pixel_score}\n"
                 f"{predictions['image_path'][0][0]}",
                 fontsize=10)

    axes[0].imshow(cv2_stacked_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    axes[1].imshow(heat_map)
    axes[1].axis("off")
    axes[1].set_title("Anomaly Heatmap")

    axes[2].imshow(pred_mask, interpolation='nearest')
    axes[2].axis("off")
    axes[2].set_title(f"Predicted segments: total predictions-->{total_preds}")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    filename="prediction.png"
    output_file = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_file)
    print(f"Saved output to {output_file}")


def plot_single_prediction(predictions, index,op_dir):
    batch_size = len(predictions[0]['pred_scores'])
    batch_idx = index // batch_size
    sample_idx = index % batch_size
    visualiser(predictions[batch_idx], sample_idx, op_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Patchcore inference with custom inputs.")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory (e.g. /exp/model_split_per_partnorm)")
    parser.add_argument("--image_path", required=True, help="Path to the input image directory")
    args = parser.parse_args()

    results,dir_op = run_pipeline(model_path=args.model_dir, image_input_path=args.image_path)

    plot_single_prediction(results, 0,"/home/experiments/imgs/predictions")  # You can change the index for different samples

    clear_split_image_directory(dir_op)
