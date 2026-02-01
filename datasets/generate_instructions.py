# generate_instructions.py
import json
import openai  # pip install openai
import time
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # set this


def make_prompts(sample):
    prompts = []
    caption = sample["caption"]
    bboxes = sample["bboxes"]
    # Format symbolic repr
    symbol_text = f"Caption: {caption}\n"
    if bboxes:
        for i, b in enumerate(bboxes):
            symbol_text += f"Lesion{i+1}: bbox={b['bbox']}, area={b['area']}\n"

    # Conversation style
    conv_prompt = f"""You are a radiology assistant. Given the symbolic description below (de-identified), produce a short dialog between a clinician and assistant focusing on findings and recommended next steps. ALWAYS include 'Recommendation' and 'Follow-up' lines. Provide concise, clinically appropriate language.

    {symbol_text}

    Dialog:"""
    # prompts.append(("conversation", conv_prompt))

    # Detailed description
    det_prompt = f"""You are a radiology reporting assistant. Given the symbolic description below, produce a detailed ultrasound report paragraph describing the lesion(s), size estimate, echotexture, likely BI-RADS impression (use conservative wording e.g., 'indeterminate'), and recommended follow-up. Always prefix with 'Report:'.

    {symbol_text}

    Report:"""
    # prompts.append(("detailed", det_prompt))

    # Complex reasoning
    cr_prompt = f"""You are a clinical reasoning assistant. Based on the symbolic description below, suggest differential diagnoses (two or three), reasoning for each, and what additional imaging or biopsy recommendations you'd make. Include the reasoning steps.

   {symbol_text}
   
   Response:"""
    # prompts.append(("complex", cr_prompt))

    # 添加位置描述提示，明确输出格式要求
    loc_prompt = f"""You are a radiology assistant. Given the symbolic description below, describe the lesion's location in the image using relative positioning (e.g., upper left, lower right) and provide bounding box coordinates. 

    Please follow this exact format:
    The lesion is located in the [relative position] part of the image. Its bounding box has coordinates (x=[x], y=[y], width=[width], height=[height]), and it covers approximately [percentage]% of the image.

    {symbol_text}

    Location description:"""
    prompts.append(("location", loc_prompt))

    return prompts
import os
from openai import OpenAI

def call_test(prompt,max_tokens=100):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-d302b6e36ff7442197787f7f6fe83ac5",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        # model="Qwen3-8B",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return completion.choices[0].message.content
    # print(completion.model_dump_json())

def call_gpt(prompt, model="gpt-4-xxl", max_tokens=500):
    # choose model available to you; adapt fields for your client
    for _ in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4.0",  # adapt to your access
                messages=[{"role":"system","content":"You are a helpful assistant."},
                          {"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("API error:", e)
            time.sleep(2)
    return ""

def generate_all(samples_json, out_jsonl):
    with open(samples_json) as f:
        samples = json.load(f)
    samples_list = []
    count = 0
    for s in samples:
        prompts = make_prompts(s)
        for ptype, prompt in prompts:
            out = call_test(prompt)
            print(f"处理第{count+1}张图片: " + out)
            count += 1
            if not out:
                continue
            samples_list.append({
                "image_id": s["image_id"],
                "symbol": s["caption"],
                "bboxes": s["bboxes"],
                # "type": ptype,
                # "instruction": prompt,
                "answer": out
             })
    with open(out_jsonl, "w", encoding="utf-8") as f:
        json.dump(samples_list, f, indent=2)


if __name__ == "__main__":
    # json_path = "D:\\HCMNet\\TextSAM-US\\data\\BUSI\\symbols_test.json"
    # ins_path = "D:\\HCMNet\\TextSAM-US\\data\\BUSI\\instructions.json"

    json_path = "./data/BUSI/symbols_test.json"
    ins_path = "./data/BUSI/instructions.json"
    generate_all(json_path, ins_path)

    # openai_api_key = os.environ.get("OPENAI_API_KEY")
    # print("OpenAI API Key:", openai_api_key)
