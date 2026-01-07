import torch
import cv2
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. 核心转换函数 (保持之前的修复版本)
def vpt_reshape_transform(tensor, height=14, width=14):
    # 只切片取出图片部分：从 1 到 1 + 196 (忽略 [0]CLS 和 [197:]Prompt)
    num_patches = height * width
    result = tensor[:, 1 : 1 + num_patches, :] 
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 2. 反归一化
def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().detach().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# 3. 主可视化函数 (已添加预测判断逻辑)
def run_vpt_visualization(model, data_loader, device, epoch, output_dir, sample_num=20):
    print(f"Start Visualizing for Epoch {epoch}...")
    model.eval()
    
    # 目标层：最后一个 Block 的 norm1
    target_layers = [model.blocks[-1].norm1]

    try:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=vpt_reshape_transform)
    except Exception as e:
        print(f"GradCAM 初始化失败: {e}")
        return

    # 获取数据
    try:
        images, labels = next(iter(data_loader))
        images = images.to(device)
        labels = labels.to(device)
    except StopIteration:
        print("DataLoader 为空")
        return

    real_sample_num = min(sample_num, images.shape[0])
    images = images[:real_sample_num]
    labels = labels[:real_sample_num]
    
    # === 新增：获取模型预测结果 ===
    # 注意：GradCAM 会调用 backward，所以这里最好用 no_grad 做一次纯前向推理拿 label
    with torch.no_grad():
        outputs = model(images)
        # 假设输出是 logits，取最大值索引
        predictions = torch.argmax(outputs, dim=1) 
    # ==========================

    # 生成热图
    grayscale_cams = cam(input_tensor=images, targets=None) # targets=None 表示解释预测值最高的类

    save_path = os.path.join(output_dir, f"vis_epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)

    for i in range(real_sample_num):
        # 1. 基础图像处理
        rgb_img = denormalize_image(images[i])
        grayscale_cam = grayscale_cams[i, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # 转为 uint8 以便 OpenCV 处理
        rgb_img_uint8 = (rgb_img * 255).astype(np.uint8)
        # 转为 BGR (OpenCV 格式)
        rgb_img_bgr = cv2.cvtColor(rgb_img_uint8, cv2.COLOR_RGB2BGR)
        vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        # 2. 拼接
        combined = np.hstack((rgb_img_bgr, vis_bgr))
        
        # 3. === 新增：添加文字标注 ===
        gt_label = labels[i].item()
        pred_label = predictions[i].item()
        
        if gt_label == pred_label:
            status = "Correct"
            color = (0, 255, 0) # 绿色 (BGR)
        else:
            status = "Wrong"
            color = (0, 0, 255) # 红色 (BGR)
            
        text = f"GT:{gt_label} Pred:{pred_label} [{status}]"
        
        # 在图片左上角写字 (位置: x=10, y=20)
        cv2.putText(combined, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2, cv2.LINE_AA)
        # ===========================

        # 4. 保存 (文件名也加上状态，方便排序)
        file_name = f'sample_{i}_{status}.jpg'
        cv2.imwrite(os.path.join(save_path, file_name), combined)
    
    print(f"Saved {real_sample_num} images with annotations to {save_path}")