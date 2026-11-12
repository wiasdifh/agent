from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 新增：用于托管前端网页和图片
from pydantic import BaseModel
import json
import os
from openai import OpenAI
from lirards_decoder import decode_lirards

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ⚠️ 记得在这里填上你的 DeepSeek API Key
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

def generate_question_llm(missing_fields, data):
    fallback = {"washout": "延迟期是否有轻度廓清？", "aphe": "动脉期是否有高强化？", "afp": "AFP指标是否升高？"}
    try:
        prompt = f"你是放射科医生。病灶大小{data['lesion']['size_mm']}mm, 缺失信息：{missing_fields}。请提一个封闭式问题（A/B/C）。"
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=100)
        return res.choices[0].message.content.strip(), missing_fields[0]
    except Exception as e:
        print(f"❌ 调用大模型提问失败: {e}")  # 这行会在终端打印真实原因
        return fallback.get(missing_fields[0], "请补充缺失信息"), missing_fields[0]

def generate_final_report_llm(lr, conf, reason):
    try:
        prompt = f"临床肝病专家。系统分级：{lr}，置信度：{conf:.2f}。依据：{reason}。请写一段包含诊断结论、随访建议、免责声明的专业报告。"
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=250)
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 调用大模型生成报告失败: {e}")  # 这行会在终端打印真实原因
        return f"诊断结论：{lr}。建议随访或MDT会诊。（离线模式）"

@app.get("/api/cases")
def get_cases():
    if not os.path.exists("static/mock_cases"): 
        return {"cases": []}
    return {"cases": [f for f in os.listdir("static/mock_cases") if f.endswith(".json")]}
# ================= 核心接口 1：初次分析 =================
class CaseRequest(BaseModel):
    case_file: str

@app.post("/api/analyze_step1")
def analyze_step1(req: CaseRequest):
    try:
        with open(f"static/mock_cases/{req.case_file}", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"读取失败: {str(e)}"}

    result = decode_lirards(data)
    
    # 日志带角色标签
    logs = [
        {"role": "system", "text": f"开始处理: {req.case_file}"},
        {"role": "inference", "text": f"[第 1 轮推理] 分级: {result['lr_category']} (置信度: {result['overall_confidence']:.2f})"}
    ]

    if result['lr_category'] in ["LR-5", "LR-4", "LR-3"] and not result['has_conflict']:
        logs.append({"role": "system", "text": "诊断明确，正在生成大模型报告..."})
        report = generate_final_report_llm(result['lr_category'], result['overall_confidence'], result['reasoning_trace'])
        return {
            "status": "done", "logs": logs, "report": report, 
            "result_data": {"lr": result['lr_category'], "conf": result['overall_confidence']},
            "evidence": json.dumps(data, ensure_ascii=False)
        }
    else:
        missing = result['missing_or_uncertain_fields']
        if not missing:
            report = generate_final_report_llm("uncertain", result['overall_confidence'], "特征不典型")
            return {"status": "done", "logs": logs, "report": report, "result_data": {"lr": "uncertain", "conf": result['overall_confidence']}, "evidence": json.dumps(data, ensure_ascii=False)}
        
        q, target = generate_question_llm(missing, data)
        logs.append({"role": "system", "text": f"⚠️ 发现证据冲突/缺失，发起追问。"})
        return {
            "status": "need_more_info", "logs": logs, "question": q, "target_field": target, 
            "round": 1, "result_data": {"lr": "uncertain", "conf": result['overall_confidence']},
            "evidence": json.dumps(data, ensure_ascii=False)
        }

# ================= 核心接口 2：接收回答 =================
class AnswerRequest(BaseModel):
    case_file: str
    target_field: str
    answer: str
    round_num: int

@app.post("/api/analyze_step2")
def analyze_step2(req: AnswerRequest):
    with open(f"static/mock_cases/{req.case_file}", "r", encoding="utf-8") as f:
        data = json.load(f)

    if req.answer == "A": 
        if req.target_field == "afp": data["clinical_info"]["afp"] = "high"
        else:
            data["features"][req.target_field]["label"] = "positive"
            data["features"][req.target_field]["confidence"] = 0.90
    
    result = decode_lirards(data)
    logs = [
        {"role": "user", "text": f"医生回答: 选项 {req.answer}"},
        {"role": "inference", "text": f"[第 {req.round_num + 1} 轮推理] 分级: {result['lr_category']} (置信度: {result['overall_confidence']:.2f})"}
    ]

    if result['has_conflict'] and req.round_num < 2:
        missing = result['missing_or_uncertain_fields']
        q, target = generate_question_llm(missing, data)
        logs.append({"role": "system", "text": f"⚠️ 仍有缺失，继续追问。"})
        return {
            "status": "need_more_info", "logs": logs, "question": q, "target_field": target, 
            "round": req.round_num + 1, "result_data": {"lr": "uncertain", "conf": result['overall_confidence']},
            "evidence": json.dumps(data, ensure_ascii=False)
        }
    
    final_lr = result['lr_category'] if result['lr_category'] != "uncertain" else "uncertain"
    logs.append({"role": "system", "text": "推理完成，正在生成最终报告..."})
    report = generate_final_report_llm(final_lr, result['overall_confidence'], result['reasoning_trace'])
    
    return {
        "status": "done", "logs": logs, "report": report, 
        "result_data": {"lr": final_lr, "conf": result['overall_confidence']},
        "evidence": json.dumps(data, ensure_ascii=False)
    }

# ================= 新增：挂载静态文件（前端网页和图片） =================
# 注意：这行代码必须放在所有 API 路由定义的最下方！
# directory="." 表示当前目录（即 D:\agent），html=True 允许直接访问目录时默认返回 index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")