import json


def decode_lirards(data):
    features = data["features"]
    size = data["lesion"]["size_mm"]

    # 提取特征和置信度
    aphe = features["aphe"]
    washout = features["washout"]
    capsule = features["capsule"]

    # 规则1：看是否有明显的矛盾和不确定性
    missing_fields = []
    if washout["label"] == "uncertain":
        missing_fields.append("washout")
    if aphe["label"] == "uncertain":
        missing_fields.append("aphe")
    if data["clinical_info"]["afp"] == "unknown":
        missing_fields.append("afp")

    # 计算一个简易的总体置信度
    overall_conf = (aphe["confidence"] + washout["confidence"] + capsule["confidence"]) / 3.0

    # 触发追问的条件（有缺失字段且不是极低置信度）
    if len(missing_fields) > 0 and overall_conf > 0.5:
        return {
            "lr_category": "uncertain",
            "overall_confidence": overall_conf,
            "has_conflict": True,
            "missing_or_uncertain_fields": missing_fields,
            "reasoning_trace": [f"发现缺失或不确定特征: {', '.join(missing_fields)}，无法确诊。"]
        }

    # 低置信度兜底
    if overall_conf < 0.5 or size < 10.0:
        return {
            "lr_category": "LR-3",  # 随访或不确定
            "overall_confidence": overall_conf,
            "has_conflict": False,
            "missing_or_uncertain_fields": missing_fields,
            "reasoning_trace": ["病灶较小或整体置信度低，建议随访。"]
        }

    # 高置信 LR-5 规则（简化版）
    if size >= 20.0 and aphe["label"] == "positive" and washout["label"] == "positive":
        return {
            "lr_category": "LR-5",
            "overall_confidence": overall_conf,
            "has_conflict": False,
            "missing_or_uncertain_fields": [],
            "reasoning_trace": ["大病灶且APHE与washout均阳性，符合LR-5标准。"]
        }

    # 其他默认打回 LR-4
    return {
        "lr_category": "LR-4",
        "overall_confidence": overall_conf,
        "has_conflict": False,
        "missing_or_uncertain_fields": missing_fields,
        "reasoning_trace": ["特征组合未达到LR-5标准，判为LR-4。"]
    }


# 测试代码
if __name__ == "__main__":
    import os

    for file in os.listdir("mock_cases"):
        if file.endswith(".json"):
            with open(f"mock_cases/{file}", "r", encoding="utf-8") as f:
                data = json.load(f)
                result = decode_lirards(data)
                print(f"--- {file} ---")
                print(result)