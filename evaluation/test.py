import cv2
import numpy as np
import os
import natsort


def testFileName(gt_path, seg_path):
    """测试预测掩码文件夹文件与真实值掩码是否一致"""
    gt_files = natsort.natsorted(os.listdir(gt_path))
    seg_files = natsort.natsorted(os.listdir(seg_path))
    for g, s in zip(gt_files, seg_files):
        print(g, " <-> ", s)

def testOverlap():
    """测试分割掩码和真实值掩码是否重叠"""
    gt = cv2.imread('/home/cwq/MedicalDP/TextSAM-US/output/TUS/gt_masks/seed3/nodule/LORA16_SHOTS-1_NCTX4_CSCFalse_CTPend/L4-0006-6.png', 0)
    seg = cv2.imread('/home/cwq/MedicalDP/TextSAM-US/output/TUS/seg_results/seed3/nodule/LORA16_SHOTS-1_NCTX4_CSCFalse_CTPend/L4-0006-6.png', 0)

    intersection = np.logical_and(gt > 0, seg > 0)
    union = np.logical_or(gt > 0, seg > 0)

    print("Overlap pixels:", intersection.sum())

if __name__ == '__main__':
    gt_path = '/home/cwq/MedicalDP/TextSAM-US/output/TUS/gt_masks/seed3/nodule/LORA16_SHOTS-1_NCTX4_CSCFalse_CTPend'
    seg_path = '/home/cwq/MedicalDP/TextSAM-US/output/TUS/seg_results/seed3/nodule/LORA16_SHOTS-1_NCTX4_CSCFalse_CTPend'

    # testFileName('/home/cwq/MedicalDP/TextSAM-US/output/TUS/gt_masks/seed3/nodule/LORA16_SHOTS-1_NCTX4_CSCFalse_CTPend',
                # '/home/cwq/MedicalDP/TextSAM-US/output/TUS/seg_results/seed3/nodule/LORA16_SHOTS-1_NCTX4_CSCFalse_CTPend')