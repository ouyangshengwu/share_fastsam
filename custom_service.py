import cv2
import numpy as np
from ultralytics import FastSAM
import time
from flask import Flask, request, Response
from flask_cors import CORS  # 导入跨域支持
import io
from PIL import Image

# 初始化Flask应用并配置跨域
app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求，生产环境可配置具体域名

# 加载FastSAM模型（只加载一次）
model = FastSAM("FastSAM-s.pt")


def process_image(image, point=[200, 200]):
    """处理图像并返回标注后的结果"""
    # 转换PIL图像为OpenCV格式
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 执行推理
    start = time.time()
    results = model(img_bgr, device="cpu", points=[point], labels=[1])
    result = results[0]
    end = time.time()
    print(f'【处理时间】: {end - start:.4f}秒')

    # 提取原始图像和掩码
    orig_img = result.orig_img
    orig_h, orig_w = orig_img.shape[:2]
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    # 绘制标注
    annotated_img = orig_img.copy()

    if masks is not None and len(masks) > 0:
        # 调整掩码尺寸并绘制
        mask = masks[0]
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        # 绘制绿色掩码
        mask_color = np.zeros_like(annotated_img)
        mask_color[mask_resized == 1] = [0, 255, 0]
        annotated_img = cv2.addWeighted(annotated_img, 0.7, mask_color, 0.3, 0)

    # 绘制提示点和文字
    cv2.circle(annotated_img, (point[0], point[1]), 10, (0, 0, 255), -1)
    cv2.putText(annotated_img, "Prompt Point", (point[0] + 15, point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return annotated_img


@app.route('/process', methods=['POST'])
def process():
    """处理上传的图像并返回处理结果"""
    # 检查是否有文件上传
    if 'image' not in request.files:
        return "未上传图像", 400

    file = request.files['image']

    # 检查文件是否为空
    if file.filename == '':
        return "未选择图像", 400

    # 检查文件格式
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    if '.' not in file.filename or \
            file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return "不支持的文件格式，支持PNG、JPG、BMP", 400

    try:
        # 读取图像
        image = Image.open(file.stream)

        # 获取可选的提示点参数（默认[200, 200]）
        try:
            point_x = int(request.form.get('x', 200))
            point_y = int(request.form.get('y', 200))
            point = [point_x, point_y]
        except ValueError:
            point = [200, 200]
            print("提示点参数无效，使用默认值")

        # 处理图像
        processed_img = process_image(image, point)

        # 转换为JPG流
        _, buffer = cv2.imencode('.jpg', processed_img)
        io_buf = io.BytesIO(buffer)

        # 返回JPG流
        return Response(
            io_buf.getvalue(),
            mimetype='image/jpeg'
        )

    except Exception as e:
        return f"处理图像时出错: {str(e)}", 500


if __name__ == '__main__':
    # 启动服务，默认在15000端口，允许外部访问
    app.run(host='0.0.0.0', port=15000, debug=False)