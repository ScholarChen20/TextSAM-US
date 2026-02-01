import base64
import os
import re
import json
from openai import OpenAI
from typing import List, Optional, Tuple, Dict

# 设置你的 DashScope API Key
# dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your_default_key_here")
def image_to_base64(image_path: str) -> str:
    """将本地图像转为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_Qwen(prompt,image_base64,max_tokens=400):

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-d302b6e36ff7442197787f7f6fe83ac5",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # model="qwen-plus",
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": prompt},
        # ],
        model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{"role": "user","content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt},
                ]}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return completion.choices[0].message.content

def extract_breast_nodule_bbox_from_report(
        image_path: str,
        report_text: str,
        image_size: Tuple[int, int] = (256, 256),
        model_name: str = "qwen-max",
        timeout: int = 30
) -> Optional[Dict]:
    """
    从乳腺超声报告中提取结节的边界框坐标（用于 SAM 提示）。

    Args:
        report_text (str): 超声报告全文（中英文均可）
        image_size (tuple): 图像尺寸，默认 (256, 256)
        model_name (str): 使用的 Qwen 模型，如 'qwen-max', 'qwen-plus'
        timeout (int): API 超时时间（秒）

    Returns:
        dict or None: {
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": float,
            "reasoning": str
        }
    """

    width, height = image_size

    # 构建结构化 Prompt（支持中英文报告）
    prompt = f"""
你是一个医学影像 AI 助手，请根据以下乳腺超声报告，在 {width}×{height} 像素的图像中推断右乳结节的大致边界框。

【报告内容】
{report_text}

【任务要求】
1. 结合256×256像素的原始乳腺肿瘤超声图片和超声报告推断出该乳腺肿瘤的边界框。
2. “10-11点钟方向”表示以乳头为中心的钟表方位，在右乳中通常对应图像的左上区域。
3. 结节大小如“34×22mm”，请按典型超声分辨率（1mm ≈ 1.2~1.8 像素）换算为像素尺寸。
4. 结节通常平行于皮肤（横向椭圆），位于浅层至中层。
5. 输出必须是严格有效的 JSON，包含：
   - "bbox": [x_min, y_min, x_max, y_max]（整数，范围 0~{width - 1}, 0~{height - 1}）
   - "confidence": 置信度（0.0~1.0）
   - "location": 位置(英文)relative positioning (e.g., upper left, lower right)
   - "reasoning": 推理说明（中文）

【输出格式】
{{"bbox": [...], "confidence": ..., "reasoning": "..."}}
不要包含任何其他文字。
"""

    try:
        try:
            image_base64 = image_to_base64(image_path)
        except Exception as e:
            print(f"Failed to load image: {e}")
            return None

        output_text = call_Qwen(prompt=prompt,image_base64=image_base64)

        # 尝试解析 JSON（兼容可能的 Markdown 包裹）
        try:
            # 去除 ```json ... ``` 包裹
            output_text = re.sub(r'^```(?:json)?\s*', '', output_text)
            output_text = re.sub(r'\s*```$', '', output_text)
            result = json.loads(output_text)
        except json.JSONDecodeError:
            # 如果失败，尝试用更宽松的方式
            import json5
            result = json5.loads(output_text)

        # 验证 bbox 格式
        bbox = result.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Invalid bbox format")

        x_min, y_min, x_max, y_max = map(int, bbox)
        # 边界裁剪
        x_min = max(0, min(width - 1, x_min))
        y_min = max(0, min(height - 1, y_min))
        x_max = max(0, min(width - 1, x_max))
        y_max = max(0, min(height - 1, y_max))

        # 确保 x_min < x_max, y_min < y_max
        if x_min >= x_max or y_min >= y_max:
            x_mid = (x_min + x_max) // 2
            y_mid = (y_min + y_max) // 2
            x_min = max(0, x_mid - 10)
            x_max = min(width - 1, x_mid + 10)
            y_min = max(0, y_mid - 10)
            y_max = min(height - 1, y_mid + 10)

        return {
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": float(result.get("confidence", 0.7)),
            "location": str(result.get("location", "")),
            "reasoning": str(result.get("reasoning", ""))
        }

    except Exception as e:
        print(f"Error in extracting bbox: {e}")
        return None


# ========================
# 示例使用
# ========================
if __name__ == "__main__":
    sample_report = """The structure of both breasts is clear, with uneven glandular echoes. Multiple cystic nodules are observed in both breasts. The largest nodule on the right side measures approximately 22×14mm, oriented parallel to the skin, and is oval in shape. The nodule's edges are smooth and well-defined, presenting as an anechoic area internally. No obvious calcifications are observed. There is no significant change in the echo pattern behind the nodule, and the surrounding glandular structures are not distorted. The ducts around the nodule are not dilated, and there is no noticeable thickening or retraction of the overlying skin.

No significantly enlarged lymph nodes are observed in the drainage areas of the bilateral axillae.

Right breast anechoic nodule, BI-RADS category 2."""
    image_path = "./data/BUS-SZU/Breast_images/L2-0003-1.jpg"
    result = extract_breast_nodule_bbox_from_report(image_path,sample_report)
    if result:
        print("✅ Extracted BBox for SAM:")
        print(f"  bbox: {result['bbox']}")
        print(f"  confidence: {result['confidence']:.2f}")
        print(f"  location: {result['location']}")
        print(f"  reasoning: {result['reasoning']}")
    else:
        print("❌ Failed to extract bbox.")