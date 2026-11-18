"""Audit views derived from the single feature evidence engine."""
from features import evidence_version, extract_text_features, knowledge_base_info


def audit_csv_rows(audits):
    """Flatten evidence while retaining texts with no rule coverage."""
    rows = []
    base_fields = [
        "review_id", "post_id", "source_row", "source_column", "original_text", "text",
        "month", "platform", "brand", "price", "product_price", "sku_id", "sku_name",
        "comment_id", "category", "rule_version", "knowledge_base_version",
        "extractor_version", "limits",
    ]
    for audit in audits:
        base = {key: audit.get(key, "") for key in base_fields}
        base["overall_decision"] = audit.get("decision", "")
        traces = audit.get("traces", [])
        if traces:
            rows.extend({**base, **trace} for trace in traces)
        else:
            rows.append({
                **base,
                "rule_id": "",
                "aspect": "",
                "standard_tag": "",
                "evidence": "",
                "start": "",
                "end": "",
                "clause": "",
                "statement_type": "",
                "sentiment_candidate": "",
                "confidence": "",
                "basis": "",
                "polarity_basis": "",
                "negation_context": "",
                "contrast_markers": "",
                "decision": audit.get("decision", ""),
            })
    return rows


def audit_text(text):
    info = knowledge_base_info()
    traces = []
    for item in extract_text_features(text):
        traces.append({
            "rule_version": item["feature_version"],
            "knowledge_base_version": item["knowledge_base_version"],
            "extractor_version": item["extractor_version"],
            "rule_id": item["rule_id"],
            "aspect": item["feature"],
            "standard_tag": item["standard_tag"],
            "evidence": item["evidence"],
            "start": item["start"],
            "end": item["end"],
            "clause": item["clause"],
            "statement_type": item["statement_type"],
            "sentiment_candidate": item["polarity_candidate"],
            "confidence": item["confidence"],
            "basis": item["basis"],
            "polarity_basis": item["polarity_basis"],
            "negation_context": item["negation_context"],
            "contrast_markers": item["contrast_markers"],
            "decision": item["decision"],
        })
    return {
        "rule_version": evidence_version(),
        "knowledge_base_version": info["knowledge_base_version"],
        "extractor_version": info["extractor_version"],
        "traces": traces,
        "decision": (
            "存在可核查规则证据候选；仍需按复核状态确认"
            if traces
            else "规则未覆盖，无法判断；不等于没有需求"
        ),
        "limits": "唯一证据引擎的确定性规则输出；不是模型思维，不推断未出现的隐含需求。",
    }
