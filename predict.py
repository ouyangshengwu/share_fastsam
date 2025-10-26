import cv2
import numpy as np
from ultralytics import FastSAM
import time

# 1. 执行推理
model = FastSAM("FastSAM-s.pt")

start=time.time()
source = "bus.jpg"
results = model(source, device="cpu", points=[[2000, 1800]], labels=[1])
result = results[0]  # 提取单张图片结果

end=time.time()

print('【时间】:'+str(end-start))
# 2. 提取核心数据
orig_img = result.orig_img  # 原始图像（2000x3000）
orig_h, orig_w = orig_img.shape[:2]  # 原始图像的高和宽
masks = result.masks.data.cpu().numpy() if result.masks is not None else None
point = [200, 200]

# 3. 绘制标注
annotated_img = orig_img.copy()

if masks is not None and len(masks) > 0:
    # 取第一个掩码，并将其缩放至原始图像尺寸
    mask = masks[0]  # 原始掩码尺寸（例如 448x448）
    mask_resized = cv2.resize(
        mask.astype(np.uint8),  # 转为uint8类型
        (orig_w, orig_h),  # 缩放至原始图像宽高
        interpolation=cv2.INTER_NEAREST  # 保持掩码二值性
    )

    # 绘制缩放后的掩码
    mask_color = np.zeros_like(annotated_img)
    mask_color[mask_resized == 1] = [0, 255, 0]  # 绿色掩码
    annotated_img = cv2.addWeighted(annotated_img, 0.7, mask_color, 0.3, 0)

# 绘制提示点
cv2.circle(annotated_img, (point[0], point[1]), 10, (0, 0, 255), -1)
cv2.putText(annotated_img, "Prompt Point", (point[0] + 15, point[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 4. 保存结果
output_path = "annotated_bus.jpg"
cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
print(f"标注图已保存至：{output_path}")