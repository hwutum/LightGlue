import os
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d

torch.set_grad_enabled(False)

def load_stereo_calib(calib_file):
    with open(calib_file, "r") as f:
        calib = yaml.safe_load(f)
    return {k: np.array(calib[k]) for k in [
        'camera_matrix_left', 'dist_coeff_left', 'camera_matrix_right', 'dist_coeff_right',
        'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q'
    ]}

def rectify_image(image, K, D, R, P, size):
    w, h = size
    K_cv, D_cv, R_cv, P_cv = K.astype(np.float32), D.astype(np.float32), R.astype(np.float32), P.astype(np.float32)
    img_np = image.squeeze(0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    if img_np.ndim == 3 and img_np.shape[0] == 1:
        img_np = img_np[0]
    elif img_np.ndim == 3:
        img_np = np.moveaxis(img_np, 0, -1)
    map1, map2 = cv2.initUndistortRectifyMap(K_cv, D_cv, R_cv, P_cv, (w, h), cv2.CV_32FC1)
    rectified = cv2.remap(img_np, map1, map2, interpolation=cv2.INTER_LINEAR)
    if rectified.ndim == 2:
        rectified = rectified[None, ...]
    elif rectified.ndim == 3:
        rectified = np.moveaxis(rectified, -1, 0)
    rectified = rectified.astype(np.float32) / 255.0
    return torch.from_numpy(rectified).unsqueeze(0)

def to_numpy_img(img_tensor):
    img = img_tensor.squeeze(0).cpu().numpy()
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.moveaxis(img, 0, -1)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return mask

def filter_keypoints_by_mask(kpts, mask_np):
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    ys = np.clip(np.round(kpts[:, 1]).astype(int), 0, mask_np.shape[0] - 1)
    xs = np.clip(np.round(kpts[:, 0]).astype(int), 0, mask_np.shape[1] - 1)
    return mask_np[ys, xs] > 0.5

def filter_feats_by_mask(feats, keep):
    for k in ["keypoints", "descriptors", "scores", "scales"]:
        if k in feats:
            arr = feats[k]
            if isinstance(arr, torch.Tensor):
                arr = arr.squeeze(0) if arr.dim() == 3 else arr
                arr = arr[keep]
                if arr.dim() == 2:
                    arr = arr[None, ...]
                feats[k] = arr
            elif isinstance(arr, np.ndarray):
                arr = arr[keep]
                arr = torch.from_numpy(arr).to(arr.device if hasattr(arr, 'device') else 'cpu')
                if arr.dim() == 2:
                    arr = arr[None, ...]
                feats[k] = arr
    return feats

def extract_and_filter_features(image, mask, extractor, device):
    """
    提取特征点并根据掩码过滤。
    返回：过滤后的 feats、原始 kpts、掩码过滤布尔数组 keep
    """
    feats = extractor.extract(image.to(device))
    kpts = feats["keypoints"].squeeze(0)
    mask_np = mask.squeeze().cpu().numpy()
    keep = filter_keypoints_by_mask(kpts, mask_np)
    print("原始特征点数量:", kpts.shape[0])
    print("掩码内特征点数量:", np.sum(keep))
    feats = filter_feats_by_mask(feats, keep)
    return feats, kpts, keep

def extract_row_center_keypoints(mask):
    """
    对掩码mask，返回每一行掩码为1区域的中点像素坐标 (N,2)，格式为[[x, y], ...]
    """
    mask_np = mask.squeeze().cpu().numpy()  # (H, W)
    H, W = mask_np.shape
    keypoints = []
    for y in range(H):
        xs = np.where(mask_np[y] > 0.5)[0]
        if xs.size > 0:
            x_center = int(np.round(xs.mean()))
            keypoints.append([x_center, y])
    return np.array(keypoints, dtype=np.float32)  # (N, 2)

def extract_row_center_matches(mask0, mask1):
    """
    对左右掩码，返回匹配点对 (pts_left, pts_right)，y相同
    """
    kpts0 = extract_row_center_keypoints(mask0)
    kpts1 = extract_row_center_keypoints(mask1)
    # 只保留y都存在的行
    y0 = set(kpts0[:,1].astype(int))
    y1 = set(kpts1[:,1].astype(int))
    common_y = sorted(list(y0 & y1))
    pts_left = np.array([[kpts0[kpts0[:,1]==y][0][0], y] for y in common_y], dtype=np.float32)
    pts_right = np.array([[kpts1[kpts1[:,1]==y][0][0], y] for y in common_y], dtype=np.float32)
    return pts_left, pts_right

def main():
    images = Path("assets")
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    calib = load_stereo_calib(images / "stereo_parameters.yaml")

    image0 = load_image(images / "img_left_000.png")
    image1 = load_image(images / "img_right_000.png")
    h, w = image0.shape[-2:]

    image0 = rectify_image(image0, calib["camera_matrix_left"], calib["dist_coeff_left"], calib["R1"], calib["P1"], (w, h))
    image1 = rectify_image(image1, calib["camera_matrix_right"], calib["dist_coeff_right"], calib["R2"], calib["P2"], (w, h))
    cv2.imwrite("outputs/img_left_000_rectified.png", to_numpy_img(image0))
    cv2.imwrite("outputs/img_right_000_rectified.png", to_numpy_img(image1))

    mask0 = load_mask(images / "img_left_000_mask_1.png")
    mask1 = load_mask(images / "img_right_000_mask_1.png")
    mask0 = rectify_image(mask0, calib["camera_matrix_left"], calib["dist_coeff_left"], calib["R1"], calib["P1"], (w, h))
    mask1 = rectify_image(mask1, calib["camera_matrix_right"], calib["dist_coeff_right"], calib["R2"], calib["P2"], (w, h))
    cv2.imwrite("outputs/img_left_000_mask_1_rectified.png", to_numpy_img(mask0))
    cv2.imwrite("outputs/img_right_000_mask_1_rectified.png", to_numpy_img(mask1))

    img0_vis = to_numpy_img(image0)
    img1_vis = to_numpy_img(image1)

    # 选择模式
    mode = "row_center"  # "feature" 或 "row_center"

    if mode == "feature":
        # --- SuperPoint+LightGlue特征点模式 ---
        feats0, kpts0, keep0 = extract_and_filter_features(image0, mask0, extractor, device)
        feats1, kpts1, keep1 = extract_and_filter_features(image1, mask1, extractor, device)

        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        axes = viz2d.plot_images([img0_vis, img1_vis])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        plt.savefig('outputs/matches.png')

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([img0_vis, img1_vis])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        plt.savefig('outputs/keypoints.png')

        # 深度估计与可视化
        pts_left = m_kpts0.cpu().numpy()
        pts_right = m_kpts1.cpu().numpy()
        disparity = pts_left[:, 0] - pts_right[:, 0]
        points_2d = np.vstack([pts_left.T, disparity, np.ones_like(disparity)]).T
        Q = calib["Q"]
        points_3d_hom = (Q @ points_2d.T).T
        points_3d = points_3d_hom[:, :3] / points_3d_hom[:, 3:4]
        depths = points_3d[:, 2]
        print("部分深度值:", depths[:10])

        plt.figure()
        plt.hist(depths[np.isfinite(depths) & (depths > 0)], bins=50, color='royalblue')
        plt.xlabel('Depth (Z)')
        plt.ylabel('Count')
        plt.title('Depth Distribution of Matched Points')
        plt.savefig('outputs/depth_hist.png')

        # 点云可视化
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        valid = np.isfinite(depths) & (depths > 0)
        ax.scatter(points_3d[valid, 0][:1000], points_3d[valid, 1][:1000], points_3d[valid, 2][:1000], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Depth)')
        plt.title('3D Points from Stereo Matching')
        plt.savefig('outputs/points3d.png')

    elif mode == "row_center":
        # --- 行中点特征点模式 ---
        pts_left, pts_right = extract_row_center_matches(mask0, mask1)
        print("行中点特征点对数量:", pts_left.shape[0])

        # 可视化：在左右图上画出这些点
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img0_vis, cmap='gray')
        plt.scatter(pts_left[:, 0], pts_left[:, 1], s=5, c='r')
        plt.title('Row Center Keypoints (Left)')
        plt.subplot(1, 2, 2)
        plt.imshow(img1_vis, cmap='gray')
        plt.scatter(pts_right[:, 0], pts_right[:, 1], s=5, c='b')
        plt.title('Row Center Keypoints (Right)')
        plt.savefig('outputs/row_center_keypoints.png')

        axes = viz2d.plot_images([img0_vis, img1_vis])
        viz2d.plot_matches(pts_left, pts_right, color="lime", lw=0.2)
        plt.savefig('outputs/row_center_matches.png')

        # 深度估计与可视化
        disparity = pts_left[:, 0] - pts_right[:, 0]
        points_2d = np.vstack([pts_left.T, disparity, np.ones_like(disparity)]).T
        Q = calib["Q"]
        points_3d_hom = (Q @ points_2d.T).T
        points_3d = points_3d_hom[:, :3] / points_3d_hom[:, 3:4]
        depths = points_3d[:, 2]
        print("行中点部分深度值:", depths[:10])

        plt.figure()
        plt.hist(depths[np.isfinite(depths) & (depths > 0)], bins=50, color='royalblue')
        plt.xlabel('Depth (Z)')
        plt.ylabel('Count')
        plt.title('Row Center Depth Distribution')
        plt.savefig('outputs/row_center_depth_hist.png')

        # 点云可视化
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        valid = np.isfinite(depths) & (depths > 0)
        x = points_3d[valid, 0][:1000]
        y = points_3d[valid, 1][:1000]
        z = points_3d[valid, 2][:1000]
        ax.scatter(x, y, z, s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Depth)')
        plt.title('3D Points from Row Center Matching')
        # plt.savefig('outputs/row_center_points3d.png')

        # 设置xyz轴比例一致
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()

if __name__ == "__main__":
    main()