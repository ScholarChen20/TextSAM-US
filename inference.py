import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import utils.utils as utils
from datasets.dataloader import DatasetSegmentation, collate_fn
from utils.processor import Samprocessor
from segment_anything import build_sam_vit_b, build_textsam_vit_b, build_textsam_vit_h, build_textsam_vit_l
from utils.lora import LoRA_Sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import random
from utils.utils import load_cfg_from_cfg_file
import logging
from utils import utils

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument('--resume', action='store_true', help="Whether to resume training")
    parser.add_argument('--seed', type=int, default=3, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def showFig(stk_out,stk_gt):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(stk_out[0].cpu().numpy(), cmap="gray")
    plt.title("Prediction")
    plt.subplot(1, 2, 2)
    plt.imshow(stk_gt[0].cpu().numpy(), cmap="gray")
    plt.title("Ground Truth")
    plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = get_arguments()

os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}"), exist_ok=True)
logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))

logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)

if cfg.seed >= 0:
    logger.info("Setting fixed seed: {}".format(cfg.seed))
    set_random_seed(cfg.seed)

results_name = (
    f"LORA{cfg.SAM.RANK}_"
    f"SHOTS{cfg.DATASET.NUM_SHOTS}_"
    f"NCTX{cfg.PROMPT_LEARNER.N_CTX_TEXT}_"
    f"CSC{cfg.PROMPT_LEARNER.CSC}_"
    f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
)

checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best"

with torch.no_grad():
    checkpoint_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth"
    )

    classnames = cfg.PROMPT_LEARNER.CLASSNAMES  #["background", "nodule"]

    if cfg.SAM.MODEL == "vit_b":
        sam = build_textsam_vit_b(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    elif cfg.SAM.MODEL == "vit_l":
        sam = build_textsam_vit_l(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
    else:
        sam = build_textsam_vit_h(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)

    sam_lora = LoRA_Sam(sam, cfg.SAM.RANK)
    model = sam_lora.sam
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    processor = Samprocessor(model)
    dice_scores = {}
    total_dice_values = []

    for text_label in classnames[1:]:
        dice_scores[text_label] = []
        os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", text_label, results_name), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_masks", f"seed{cfg.seed}", text_label, results_name), exist_ok=True)

    dataset = DatasetSegmentation(cfg, processor, mode="test")
    test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model.eval()
    model.to(device)

    progress_bar = tqdm(test_dataloader, desc="Evaluating", dynamic_ncols=True)

    for i, batch in enumerate(progress_bar):
        outputs = model(batched_input=batch, multimask_output=False)
        stk_gt = batch[0]["ground_truth_mask"]
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)

        # stk_gt = (stk_gt > 0.5).float()
        # print("GT min:", stk_gt.min().item(), "max:", stk_gt.max().item(), "mean:", stk_gt.mean().item())
        # stk_out = (stk_out > 0.5).float()
        # print("Pred min:", stk_out.min().item(), "max:", stk_out.max().item(), "mean:", stk_out.mean().item())

        text_labels = batch[0]["text_labels"]

        all_points = []
        all_labels = []
        all_boxes = []

        for b in range(stk_gt.shape[0]):  # batch size
            mask = stk_gt[b].detach().cpu().numpy()

            labels = text_labels.detach().cpu().numpy().reshape(-1)  # 确保是1D
            pts, lbls = utils.get_centroid_points(mask, labels)
            pts_tensor = torch.tensor(pts, dtype=torch.float32).to(device)
            lbls_tensor = torch.tensor(lbls, dtype=torch.int64).to(device)
            box = utils.get_bounding_box(mask)
            box_tensor = torch.tensor(box, dtype=torch.float32).to(device)
            print("Box:", box)
            print("Point:", pts, "Label:", lbls)

            all_points.append(pts_tensor)
            all_labels.append(lbls_tensor)
            all_boxes.append(box_tensor)

        # ==== Padding points 与 labels ====
        max_points = max([p.shape[0] for p in all_points])  # 找到 batch 中最多的点数
        padded_points = []
        padded_labels = []
        for p, l in zip(all_points, all_labels):
            pad_len = max_points - p.shape[0]
            if pad_len > 0:
                p = torch.cat([p, torch.zeros(pad_len, 2, dtype=torch.float32)], dim=0)
                l = torch.cat([l, torch.full((pad_len,), -1, dtype=torch.int64)], dim=0)  # -1表示无效标签
            padded_points.append(p)
            padded_labels.append(l)
        point_coords = torch.stack(padded_points, dim=0)  # (B, N, 2)
        point_labels = torch.stack(padded_labels, dim=0)  # (B, N)
        points = (point_coords.to(device), point_labels.to(device))
        # ==== Boxes ====
        bboxes = torch.stack(all_boxes, dim=0).to(device)  # (B, 4)
        batch[0]["points"] = points
        batch[0]["boxes"] = bboxes
        # print("Boxes:", batch[0]["boxes"])  # 测试
        # print("Points:", batch[0]["points"][0].shape, batch[0]["points"][1])


        outputs = model(batched_input=batch, multimask_output=False)
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)


        all_points = []
        all_labels = []
        all_boxes = []

        for b in range(stk_out.shape[0]):  # batch size
            mask = stk_gt[b].detach().cpu().numpy()            # shape: (H, W) or (C, H, W)

            labels = text_labels.detach().cpu().numpy().reshape(-1)  # 保证是1D
            pts, lbls = utils.get_centroid_points(mask, labels)
            box = utils.get_bounding_box(mask)
            pts_tensor = torch.tensor(pts, dtype=torch.float32).to(device)
            lbls_tensor = torch.tensor(lbls, dtype=torch.int64).to(device)
            box_tensor = torch.tensor(box, dtype=torch.float32).to(device)

            all_boxes.append(box_tensor)
            all_points.append(pts_tensor)
            all_labels.append(lbls_tensor)
            print("Box:", box)
            print("Point:", pts, "Label:", lbls)

        # Stack all batch outputs
        point_coords = torch.stack(all_points)  # shape: (B, N, 2) if N same across batch
        point_labels = torch.stack(all_labels)  # shape: (B, N)
        points = point_coords, point_labels

        # Stack to shape (B, 1, 4) — [x_min, y_min, x_max, y_max] per sample
        bboxes = torch.cat(all_boxes, dim=0)  # (B, 4)

        batch[0]["points"] = points
        batch[0]["boxes"] = bboxes

        outputs = model(batched_input=batch, multimask_output=False)
        stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)
        stk_out = stk_out.float()
        for j, label in enumerate(text_labels):
            label_j = int(label.detach().cpu())
            # print("Pred max:", stk_out.max().item(), "Pred min:", stk_out.min().item(), "mean:", stk_out.mean().item())

            # stk_out = (stk_out > 0.5).float() # 阈值化
            mask_pred = (stk_out[j].detach().cpu().numpy() * 255).astype(np.uint8)
            gt_mask = (stk_gt[j].detach().cpu().numpy() * 255).astype(np.uint8)

            # mask_pred = np.uint8(stk_out[j].detach().cpu())
            # gt_mask = np.uint8(stk_gt[j].detach().cpu())
            # print("Pred shape:", stk_out.shape, "GT shape:", stk_gt.shape)   #(1,256,256) shape size

            cv2.imwrite(os.path.join(cfg.output_dir,
                                     cfg.DATASET.NAME,
                                     "seg_results",
                                     f"seed{cfg.seed}",
                                     classnames[label_j],
                                     results_name,
                                     batch[0]["mask_name"]), mask_pred * 255)

            cv2.imwrite(os.path.join(cfg.output_dir,
                                     cfg.DATASET.NAME,
                                     "gt_masks",
                                     f"seed{cfg.seed}",
                                     classnames[label_j],
                                     results_name,
                                     batch[0]["mask_name"]), gt_mask * 255)