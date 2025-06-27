import cv2
import numpy as np
import glob
import os
import yaml


def load_stereo_calib(calib_file):
    with open(calib_file, "r") as f:
        calib = yaml.safe_load(f)
    # 读取内参和外参
    K_left = np.array(calib['camera_matrix_left'])
    D_left = np.array(calib['dist_coeff_left'])
    K_right = np.array(calib['camera_matrix_right'])
    D_right = np.array(calib['dist_coeff_right'])
    R = np.array(calib['R'])
    T = np.array(calib['T'])
    R1 = np.array(calib['R1'])
    R2 = np.array(calib['R2'])
    P1 = np.array(calib['P1'])
    P2 = np.array(calib['P2'])
    Q = np.array(calib['Q'])
    # 其他参数可按需读取
    return {
        "K_left": K_left,
        "D_left": D_left,
        "K_right": K_right,
        "D_right": D_right,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q
    }

def realtime_stereo_preview(left_cam_id, right_cam_id, calib_yaml):
    # 读取标定参数
    calib = load_stereo_calib(calib_yaml)
    K_left, D_left, R1, P1 = calib["K_left"], calib["D_left"], calib["R1"], calib["P1"]
    K_right, D_right, R2, P2 = calib["K_right"], calib["D_right"], calib["R2"], calib["P2"]
    T = calib["T"]
    baseline = abs(T[0])
    focal = K_left[0, 0]

    cap_left = cv2.VideoCapture(left_cam_id)
    cap_right = cv2.VideoCapture(right_cam_id)

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 先获取一帧确定分辨率
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    if not ret_l or not ret_r:
        print("无法读取摄像头画面")
        cap_left.release()
        cap_right.release()
        return
    h, w = frame_l.shape[:2]

    # 计算去畸变和校正映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, (w, h), cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, (w, h), cv2.CV_16SC2)

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            print("无法读取摄像头画面")
            break

        # 去畸变和校正
        frame_l_rect = cv2.remap(frame_l, left_map1, left_map2, cv2.INTER_LINEAR)
        frame_r_rect = cv2.remap(frame_r, right_map1, right_map2, cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(frame_l_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(frame_r_rect, cv2.COLOR_BGR2GRAY)

        disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan  # 无效视差设为nan
        depth = focal * baseline / (disp + 1e-6)
        depth_vis = np.nan_to_num(depth, nan=0)
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 拼接显示（缩小显示，防止1080p窗口过大）
        scale = 0.2  # 可根据需要调整缩放比例
        show = np.hstack([
            cv2.resize(frame_l_rect, (int(left_gray.shape[1]*scale), int(left_gray.shape[0]*scale))),
            cv2.resize(frame_r_rect, (int(right_gray.shape[1]*scale), int(right_gray.shape[0]*scale))),
            cv2.resize(depth_vis, (int(left_gray.shape[1]*scale), int(left_gray.shape[0]*scale)))
        ])
        cv2.imshow("Left | Right | Depth", show)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def capture_images(left_cam_id, right_cam_id, save_left_dir, save_right_dir, prefix="img", max_num=20):
    """
    拍摄并保存双目图片
    """
    os.makedirs(save_left_dir, exist_ok=True)
    os.makedirs(save_right_dir, exist_ok=True)
    cap_left = cv2.VideoCapture(left_cam_id)
    cap_right = cv2.VideoCapture(right_cam_id)

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 校验实际分辨率
    actual_left_w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_left_h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_right_w = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_right_h = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Left camera actual resolution: {actual_left_w}x{actual_left_h}")
    print(f"Right camera actual resolution: {actual_right_w}x{actual_right_h}")

    idx = 0
    print("按空格拍照，按q退出。")
    while idx < max_num:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            print("无法读取摄像头画面")
            break
        show = np.hstack([frame_l, frame_r])
        cv2.imshow("Left | Right", show)
        key = cv2.waitKey(1)
        if key == ord(' '):
            left_path = os.path.join(save_left_dir, f"{prefix}_left_{idx:03d}.png")
            right_path = os.path.join(save_right_dir, f"{prefix}_right_{idx:03d}.png")
            cv2.imwrite(left_path, frame_l)
            cv2.imwrite(right_path, frame_r)
            print(f"已保存: {left_path}, {right_path}")
            idx += 1
        elif key == ord('q'):
            break
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def stereo_calibrate(
    left_images_dir, right_images_dir,
    board_size=(11, 8), square_size=0.006,
    output_file='assets/stereo_parameters.yaml',
    visualize=True  # 新增参数
):
    """
    针对具有挑战性的双目设置（窄FOV、非平行、小重叠）的鲁棒标定程序。

    流程:
    1. 为左右相机分别加载图像。
    2. 准备棋盘格的世界坐标 (object points)。
    3. (可选但强烈推荐) 分别对左右相机进行单目标定，获得高精度的内参。
    4. 使用获得的精确内参作为输入，进行立体标定，并使用 cv2.CALIB_FIX_INTRINSIC 标志
       来固定内参，让算法专注于求解外参 (R, T)。
    5. 保存所有标定参数。

    Args:
        left_images_dir (str): 左相机图像文件夹路径。
        right_images_dir (str): 右相机图像文件夹路径。
        board_size (tuple): 棋盘格内部角点的数量 (width, height)。
        square_size (float): 棋盘格方块的边长（米）。
        output_file (str): 输出YAML文件的路径。
    """
    print("标定流程开始...")

    # 1. 准备工作
    # --------------------------------------------------------------------
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 准备世界坐标系中的点, (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # 存储检测到的角点的世界坐标和图像坐标
    objpoints = []  # 3d point in real world space
    imgpoints_l = []  # 2d points in image plane for left camera
    imgpoints_r = []  # 2d points in image plane for right camera

    # 加载图像
    left_images = sorted(glob.glob(os.path.join(left_images_dir, '*.png')))
    right_images = sorted(glob.glob(os.path.join(right_images_dir, '*.png')))

    if len(left_images) == 0 or len(right_images) == 0:
        print("错误：在指定目录中未找到图像。请检查路径。")
        return
    if len(left_images) != len(right_images):
        print("警告：左右相机图像数量不匹配。仅使用匹配的图像对。")
        # 简单处理，可以根据文件名进行更复杂的匹配
        num_images = min(len(left_images), len(right_images))
        left_images = left_images[:num_images]
        right_images = right_images[:num_images]

    img_shape = None

    # 2. 提取角点
    # --------------------------------------------------------------------
    print(f"开始从 {len(left_images)} 对图像中提取角点...")
    for i, (fname_l, fname_r) in enumerate(zip(left_images, right_images)):
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)
        
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray_l.shape[::-1]

        # 寻找棋盘格角点
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, None)

        # 可视化角点检测结果
        if visualize:
            vis_l = img_l.copy()
            vis_r = img_r.copy()
            cv2.drawChessboardCorners(vis_l, board_size, corners_l, ret_l)
            cv2.drawChessboardCorners(vis_r, board_size, corners_r, ret_r)
            show = np.hstack([vis_l, vis_r])
            show = cv2.resize(show, (show.shape[1]//2, show.shape[0]//2))
            cv2.imshow("Corners Visualization", show)
            key = cv2.waitKey(500)  # 500ms自动切换，或按任意键跳过
            if key == 27:  # ESC退出可视化
                visualize = False
                cv2.destroyWindow("Corners Visualization")

        # 如果左右图像都成功找到了角点
        if ret_l and ret_r:
            objpoints.append(objp)
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners2_l)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_r.append(corners2_r)
            print(f"  图像对 {i+1}/{len(left_images)}: 成功找到角点。")
        else:
            print(f"  图像对 {i+1}/{len(left_images)}: 未能同时找到角点，跳过此对。")
    
    if len(objpoints) < 10:
        print(f"错误：有效的图像对太少 ({len(objpoints)}), 无法进行标定。建议至少有20对有效图像。")
        return

    print(f"\n角点提取完成。共找到 {len(objpoints)} 对有效角点。")

    # 3. (推荐步骤) 单独进行单目标定以获得精确内参
    # --------------------------------------------------------------------
    # 注意：这里我们使用双目标定的角点数据进行单目标定。
    # 为了达到最佳效果，应按照第2步A节的建议，使用单独拍摄的单目图像集进行此步骤。
    # 这里为了代码简洁，我们暂时复用双目数据。
    print("\n步骤 3: 单独标定每个相机以内参...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)
    print("单目标定完成。")


    # 4. 执行立体标定
    # --------------------------------------------------------------------
    print("\n步骤 4: 执行立体标定...")
    
    # 设置标定标志
    # 这是最关键的一步:
    # - cv2.CALIB_FIX_INTRINSIC: 固定内参矩阵和畸变系数，让优化算法全力求解 R 和 T。
    # - cv2.CALIB_USE_INTRINSIC_GUESS: 将我们上一步算出的高精度内参作为初始值。
    # - cv2.CALIB_RATIONAL_MODEL: 如果你的相机畸变很复杂（窄FOV通常不至于），可以启用。
    stereo_flags = cv2.CALIB_FIX_INTRINSIC

    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        mtx_l,       # 使用上一步计算的左相机内参作为初始值
        dist_l,      # 使用上一步计算的左相机畸变作为初始值
        mtx_r,       # 使用上一步计算的右相机内参作为初始值
        dist_r,      # 使用上一步计算的右相机畸变作为初始值
        img_shape,
        criteria=criteria,
        flags=stereo_flags
    )

    if ret:
        print(f"立体标定成功！最终重投影误差: {ret}")
    else:
        print("立体标定失败。")
        return

    # 5. 计算校正参数和投影矩阵
    # --------------------------------------------------------------------
    print("\n步骤 5: 计算校正变换...")
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape,
        R, T,
        alpha=0.9 # alpha=0 保留所有像素但有黑色区域, alpha=1 裁剪掉所有因校正产生的无效像素
    )
    print("校正变换计算完成。")


    # 6. 保存结果
    # --------------------------------------------------------------------
    print(f"\n步骤 6: 保存参数到 {output_file}...")
    data = {
        'camera_matrix_left': mtx_l.tolist(),
        'dist_coeff_left': dist_l.tolist(),
        'camera_matrix_right': mtx_r.tolist(),
        'dist_coeff_right': dist_r.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        'R1': R1.tolist(),
        'R2': R2.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'Q': Q.tolist(),
        'image_size': img_shape,
        'valid_roi_left': validPixROI1,
        'valid_roi_right': validPixROI2
    }
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("\n标定流程全部完成！")

    # 7. (验证步骤) 可视化校正效果
    # --------------------------------------------------------------------
    print("\n步骤 7: 生成并显示校正效果预览...")
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)

    # 选择一对图像进行预览
    img_l_orig = cv2.imread(left_images[3])
    img_r_orig = cv2.imread(right_images[3])

    img_l_rectified = cv2.remap(img_l_orig, map1_l, map2_l, cv2.INTER_LINEAR)
    img_r_rectified = cv2.remap(img_r_orig, map1_r, map2_r, cv2.INTER_LINEAR)

    # 将两张校正后的图像并排显示
    combined_image = np.hstack([img_l_rectified, img_r_rectified])

    # 在图像上绘制水平线以检查对齐情况
    for i in range(20, combined_image.shape[0], 50):
        cv2.line(combined_image, (0, i), (combined_image.shape[1], i), (0, 255, 0), 1)

    # 缩小图像以便显示
    h, w = combined_image.shape[:2]
    scale_factor = 1280 / w # 调整宽度到1280像素
    small_combined_image = cv2.resize(combined_image, (int(w*scale_factor), int(h*scale_factor)))

    cv2.imshow('Rectified Images with Epipolar Lines', small_combined_image)
    print("按任意键退出预览。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def chessboard_homography_warp(left_img_path, right_img_path, board_size=(11, 8), visualize=True):
    """
    检测左右图像的棋盘格角点，使用角点进行配对，估计单应矩阵，
    并将右图通过单应变换投影到左图视角下进行可视化。
    """
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, None)

    if not (ret_l and ret_r):
        print("未能在两张图像中都找到棋盘格角点，无法进行单应估计。")
        return None

    # 亚像素精确化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

    # 计算单应矩阵 H: right -> left
    H, mask = cv2.findHomography(corners_r, corners_l, cv2.RANSAC)

    # 投影右图到左图视角
    warped_r = cv2.warpPerspective(img_r, H, (img_l.shape[1], img_l.shape[0]))

    if visualize:
        # 可视化：左图、右图、投影后的右图
        stacked = np.hstack([
            img_l,
            img_r,
            warped_r
        ])
        cv2.imshow("Left | Right | Right Warped to Left", stacked)
        print("按任意键关闭窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return H, warped_r



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='双目相机标定')
    parser.add_argument('--left_dir', type=str, default="assets/left_narrow", help='左相机图像文件夹')
    parser.add_argument('--right_dir', type=str, default="assets/right_narrow", help='右相机图像文件夹')
    parser.add_argument('--board_w', type=int, default=11, help='棋盘格内角点宽度')
    parser.add_argument('--board_h', type=int, default=8, help='棋盘格内角点高度')
    parser.add_argument('--square_size', type=float, default=0.006, help='棋盘格方格边长，m')
    parser.add_argument('--output', type=str, default='outputs/stereo_parameters_narrow.yaml', help='输出文件名')
    parser.add_argument('--capture', action='store_true', help='是否进入拍摄模式')
    parser.add_argument('--preview', action='store_true', help='是否实时预览深度')
    parser.add_argument('--left_cam', type=int, default=10, help='左相机ID')
    parser.add_argument('--right_cam', type=int, default=12, help='右相机ID')
    parser.add_argument('--max_num', type=int, default=20, help='最多拍摄图片对数')
    parser.add_argument('--prefix', type=str, default='img', help='保存图片前缀')
    args = parser.parse_args()

    if args.capture:
        capture_images(
            left_cam_id=args.left_cam,
            right_cam_id=args.right_cam,
            save_left_dir=args.left_dir,
            save_right_dir=args.right_dir,
            prefix=args.prefix,
            max_num=args.max_num
        )
    elif args.preview:
        realtime_stereo_preview(
            left_cam_id=args.left_cam,
            right_cam_id=args.right_cam,
            calib_yaml=args.output
        )
    else:
        stereo_calibrate(
            args.left_dir, args.right_dir,
            board_size=(args.board_w, args.board_h),
            square_size=args.square_size,
            output_file=args.output
        )
    chessboard_homography_warp(
        left_img_path="assets/left_narrow/img_left_000.png",
        right_img_path="assets/right_narrow/img_right_000.png",
        board_size=(11, 8),
        visualize=True
    )
