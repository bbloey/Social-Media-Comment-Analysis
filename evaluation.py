"""Sampling and gold-standard metrics for deterministic feature evidence."""
import random

import pandas as pd


GOLD_COLUMNS = [
    "review_id", "text", "annotation_complete", "gold_no_feature", "gold_feature",
    "gold_standard_tag", "gold_evidence", "gold_start", "gold_end", "gold_polarity",
    "gold_statement_type", "annotator", "adjudication_status", "notes",
]


def _sample(values, size, rng):
    values = sorted(set(str(value) for value in values))
    return rng.sample(values, min(int(size), len(values))) if values else []


def build_review_samples(records, feature_rows, n_hits=20, n_uncovered=20, seed=42):
    record_map = {str(row["review_id"]): row for row in records}
    predictions = [row for row in feature_rows if row.get("source") == "文本规则"]
    hit_ids = {str(row["review_id"]) for row in predictions}
    all_ids = set(record_map)
    rng = random.Random(seed)
    sampled_hits = _sample(hit_ids & all_ids, n_hits, rng)
    sampled_uncovered = _sample(all_ids - hit_ids, n_uncovered, rng)

    quick_hits = []
    for row in predictions:
        review_id = str(row["review_id"])
        if review_id not in sampled_hits:
            continue
        quick_hits.append({
            "review_id": review_id,
            "text": record_map[review_id].get("text", ""),
            "rule_id": row.get("rule_id", ""),
            "feature": row.get("feature", ""),
            "standard_tag": row.get("standard_tag", ""),
            "evidence": row.get("evidence", ""),
            "polarity": row.get("polarity_candidate", ""),
            "statement_type": row.get("statement_type", ""),
            "review_result": "未复核",
            "corrected_feature": "",
            "corrected_tag": "",
            "corrected_polarity": "",
            "review_notes": "",
        })
    quick_uncovered = [{
        "review_id": review_id,
        "text": record_map[review_id].get("text", ""),
        "coverage_review": "未复核",
        "missed_feature": "",
        "missed_evidence": "",
        "review_notes": "",
    } for review_id in sampled_uncovered]

    gold_template = []
    for review_id in sampled_hits + sampled_uncovered:
        gold_template.append({
            "review_id": review_id,
            "text": record_map[review_id].get("text", ""),
            "annotation_complete": False,
            "gold_no_feature": False,
            "gold_feature": "",
            "gold_standard_tag": "",
            "gold_evidence": "",
            "gold_start": "",
            "gold_end": "",
            "gold_polarity": "",
            "gold_statement_type": "",
            "annotator": "",
            "adjudication_status": "未仲裁",
            "notes": "每个真实证据一行；同一评论有多个证据时复制 review_id 和原文新增行",
        })
    return {
        "hit_review_ids": sampled_hits,
        "uncovered_review_ids": sampled_uncovered,
        "quick_hits": quick_hits,
        "quick_uncovered": quick_uncovered,
        "gold_template": gold_template,
    }


def quick_review_metrics(hit_rows, uncovered_rows):
    hit = pd.DataFrame(hit_rows)
    uncovered = pd.DataFrame(uncovered_rows)
    reviewed_hits = hit[hit.get("review_result", pd.Series(dtype=str)).isin(["正确", "错误"])] if not hit.empty else hit
    reviewed_uncovered = uncovered[
        uncovered.get("coverage_review", pd.Series(dtype=str)).isin(["有遗漏", "无遗漏"])
    ] if not uncovered.empty else uncovered
    return {
        "reviewed_evidence": len(reviewed_hits),
        "estimated_precision": (
            float((reviewed_hits["review_result"] == "正确").mean()) if len(reviewed_hits) else None
        ),
        "reviewed_uncovered_reviews": len(reviewed_uncovered),
        "uncovered_demand_discovery_rate": (
            float((reviewed_uncovered["coverage_review"] == "有遗漏").mean())
            if len(reviewed_uncovered) else None
        ),
        "is_formal_recall": False,
    }


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _normalize_gold(gold_rows, sampled_review_ids):
    rows = pd.DataFrame(gold_rows).fillna("").to_dict("records")
    sample_ids = {str(value) for value in sampled_review_ids}
    rows = [row for row in rows if str(row.get("review_id", "")) in sample_ids]
    grouped = {review_id: [] for review_id in sample_ids}
    for row in rows:
        grouped[str(row["review_id"])].append(row)
    incomplete = [review_id for review_id, group in grouped.items()
        if not group or not all(_truthy(row.get("annotation_complete")) for row in group)]
    if incomplete:
        raise ValueError("以下样本尚未完成完整标注：" + "、".join(sorted(incomplete)[:20]))

    gold = []
    for row in rows:
        no_feature = _truthy(row.get("gold_no_feature"))
        feature = str(row.get("gold_feature", "")).strip()
        evidence = str(row.get("gold_evidence", "")).strip()
        if no_feature:
            if feature or evidence:
                raise ValueError(f"评论 {row['review_id']} 同时标记无特征和证据")
            continue
        if not feature or not evidence:
            raise ValueError(f"评论 {row['review_id']} 缺少 gold_feature 或 gold_evidence；无特征请勾选 gold_no_feature")
        text = str(row.get("text", ""))
        try:
            start = int(row.get("gold_start", ""))
            end = int(row.get("gold_end", ""))
        except (TypeError, ValueError):
            if text.count(evidence) != 1:
                raise ValueError(f"评论 {row['review_id']} 的证据重复或不存在，请填写准确起止位置")
            start = text.index(evidence)
            end = start + len(evidence)
        if start < 0 or end <= start or text[start:end] != evidence:
            raise ValueError(f"评论 {row['review_id']} 的证据和起止位置不一致")
        gold.append({
            "review_id": str(row["review_id"]),
            "feature": feature,
            "standard_tag": str(row.get("gold_standard_tag", "")).strip(),
            "evidence": evidence,
            "start": start,
            "end": end,
            "polarity": str(row.get("gold_polarity", "")).strip(),
            "statement_type": str(row.get("gold_statement_type", "")).strip(),
        })
    return gold


def _overlap(left, right):
    return max(0, min(int(left["end"]), int(right["end"])) - max(int(left["start"]), int(right["start"])))


def evaluate_gold_standard(feature_rows, gold_rows, sampled_review_ids):
    sample_ids = {str(value) for value in sampled_review_ids}
    predictions = [{
        "review_id": str(row.get("review_id", "")),
        "feature": str(row.get("feature", "")),
        "standard_tag": str(row.get("standard_tag", "")),
        "evidence": str(row.get("evidence", "")),
        "start": int(row.get("start", 0)),
        "end": int(row.get("end", 0)),
        "polarity": str(row.get("polarity_candidate", "")),
        "statement_type": str(row.get("statement_type", "")),
        "rule_id": str(row.get("rule_id", "")),
    } for row in feature_rows if row.get("source") == "文本规则" and str(row.get("review_id", "")) in sample_ids]
    gold = _normalize_gold(gold_rows, sample_ids)

    candidates = []
    for pred_index, pred in enumerate(predictions):
        for gold_index, truth in enumerate(gold):
            if pred["review_id"] != truth["review_id"] or pred["feature"] != truth["feature"]:
                continue
            overlap = _overlap(pred, truth)
            if not overlap:
                continue
            exact_span = pred["start"] == truth["start"] and pred["end"] == truth["end"]
            tag_match = pred["standard_tag"] == truth["standard_tag"]
            score = (4 if exact_span else 0) + (2 if tag_match else 0) + overlap / max(1, truth["end"] - truth["start"])
            candidates.append((score, pred_index, gold_index, exact_span, tag_match))
    matches = []
    used_predictions = set()
    used_gold = set()
    for score, pred_index, gold_index, exact_span, tag_match in sorted(candidates, reverse=True):
        if pred_index in used_predictions or gold_index in used_gold:
            continue
        used_predictions.add(pred_index)
        used_gold.add(gold_index)
        matches.append((pred_index, gold_index, exact_span, tag_match))

    tp = len(matches)
    precision = tp / len(predictions) if predictions else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not predictions else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(bool(item[2]) for item in matches)
    tag_matches = sum(bool(item[3]) for item in matches)
    polarity_pairs = [(predictions[p]["polarity"], gold[g]["polarity"]) for p, g, _, _ in matches if gold[g]["polarity"]]
    statement_pairs = [(predictions[p]["statement_type"], gold[g]["statement_type"])
        for p, g, _, _ in matches if gold[g]["statement_type"]]
    return {
        "annotation_unit": "evidence_span",
        "sampled_reviews": len(sample_ids),
        "predicted_evidence": len(predictions),
        "gold_evidence": len(gold),
        "true_positive_relaxed": tp,
        "precision_relaxed": precision,
        "recall_relaxed": recall,
        "f1_relaxed": f1,
        "exact_span_rate_among_matches": exact / tp if tp else None,
        "tag_accuracy_among_matches": tag_matches / tp if tp else None,
        "polarity_accuracy_given_matched_feature": (
            sum(left == right for left, right in polarity_pairs) / len(polarity_pairs) if polarity_pairs else None
        ),
        "statement_type_accuracy_given_matched_feature": (
            sum(left == right for left, right in statement_pairs) / len(statement_pairs) if statement_pairs else None
        ),
        "false_positives": [row for index, row in enumerate(predictions) if index not in used_predictions],
        "false_negatives": [row for index, row in enumerate(gold) if index not in used_gold],
    }
