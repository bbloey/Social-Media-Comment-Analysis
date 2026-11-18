"""Single knowledge-base-driven evidence engine for Chinese e-commerce reviews."""
import csv
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path


EXTRACTOR_VERSION = "extractor-2.0.0"
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_TAXONOMY_PATH = RESOURCE_DIR / "feature_taxonomy.csv"
DEFAULT_LEXICON_PATH = RESOURCE_DIR / "feature_lexicon.csv"
DEFAULT_CHANGELOG_PATH = RESOURCE_DIR / "knowledge_base_changelog.csv"
TRUTHY = {"1", "true", "yes", "y", "on"}
POLARITIES = {"正面", "负面", "中性", "待判定", "不适用"}
STATEMENT_TYPES = {"体验", "诉求", "场景", "关注", "描述", "营销", "未知", "数据标签"}
POLARITY_STRATEGIES = {"fixed", "context", "none"}
CLAUSE_BOUNDARY = re.compile(r"[，。！？；\n,!?;]")
CONTRAST = re.compile(r"但是|不过|可是|然而|但")
REQUEST = re.compile(r"希望|想要|能不能|最好|建议|要是|如果|期待|可以出|求出")
MARKETING = re.compile(r"新品上市|抢购|直播间|优惠券|点击购买|限时|下单")
NEGATION_PREFIX = re.compile(r"(?:不是|并非|没有|不觉得|没觉得|不算|没那么|没有那么|未必|不太|不)$")
NEGATION_IN_EVIDENCE = re.compile(r"^(?:不是|并非|没有|不觉得|没觉得|不算|没那么|没有那么|未必|不太)")


def _resource_path(env_name, default):
    configured = os.environ.get(env_name, "").strip()
    return Path(configured).expanduser().resolve() if configured else default


def taxonomy_path():
    return _resource_path("REVIEW_FEATURE_TAXONOMY", DEFAULT_TAXONOMY_PATH)


def lexicon_path():
    return _resource_path("REVIEW_FEATURE_LEXICON", DEFAULT_LEXICON_PATH)


def changelog_path():
    return _resource_path("REVIEW_FEATURE_CHANGELOG", DEFAULT_CHANGELOG_PATH)


def _stamp(path):
    if not path.is_file():
        raise FileNotFoundError(f"Feature knowledge-base file not found: {path}")
    return path.stat().st_mtime_ns


@lru_cache(maxsize=8)
def _load_taxonomy_cached(path_text, stamp):
    del stamp
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("enabled", "").strip().lower() in TRUTHY]
    required = {"order", "enabled", "feature", "definition"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Taxonomy must contain columns: {sorted(required)}")
    for row in rows:
        row["order"] = int(row["order"])
        row["feature"] = row["feature"].strip()
        row["definition"] = row["definition"].strip()
    rows.sort(key=lambda row: row["order"])
    features = [row["feature"] for row in rows]
    if any(not item for item in features) or len(features) != len(set(features)):
        raise ValueError("Feature taxonomy contains blank or duplicate feature names")
    if len(features) > 10:
        raise ValueError("Enabled top-level features exceed the configured limit of 10")
    return tuple(rows)


def load_taxonomy():
    path = taxonomy_path()
    return list(_load_taxonomy_cached(str(path), _stamp(path)))


@lru_cache(maxsize=8)
def _load_rules_cached(path_text, stamp, taxonomy_signature):
    del stamp
    allowed = set(taxonomy_signature)
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("enabled", "").strip().lower() in TRUTHY]
    required = {
        "enabled", "priority", "rule_id", "feature", "standard_tag", "pattern",
        "default_polarity", "statement_type", "polarity_strategy", "positive_context",
        "negative_context", "neutral_context", "basis", "examples", "version_note", "notes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Lexicon must contain columns: {sorted(required)}")
    seen = set()
    text_fields = required - {"enabled", "priority"}
    for row in rows:
        row["priority"] = int(row["priority"])
        for field in text_fields:
            row[field] = row.get(field, "").strip()
        if not row["rule_id"] or row["rule_id"] in seen:
            raise ValueError(f"Blank or duplicate rule_id: {row['rule_id']!r}")
        seen.add(row["rule_id"])
        if row["feature"] not in allowed:
            raise ValueError(f"Rule {row['rule_id']} references unknown feature: {row['feature']}")
        if row["default_polarity"] not in POLARITIES:
            raise ValueError(f"Rule {row['rule_id']} has invalid polarity: {row['default_polarity']}")
        if row["statement_type"] not in STATEMENT_TYPES:
            raise ValueError(f"Rule {row['rule_id']} has invalid statement_type: {row['statement_type']}")
        if row["polarity_strategy"] not in POLARITY_STRATEGIES:
            raise ValueError(f"Rule {row['rule_id']} has invalid polarity_strategy: {row['polarity_strategy']}")
        for field in ["pattern", "positive_context", "negative_context", "neutral_context"]:
            if not row[field]:
                continue
            try:
                re.compile(row[field])
            except re.error as exc:
                raise ValueError(f"Rule {row['rule_id']} has invalid {field} regex: {exc}") from exc
    rows.sort(key=lambda row: (row["priority"], row["rule_id"]))
    return tuple(rows)


def load_rules():
    path = lexicon_path()
    taxonomy = load_taxonomy()
    return list(_load_rules_cached(str(path), _stamp(path), tuple(row["feature"] for row in taxonomy)))


@lru_cache(maxsize=8)
def _knowledge_base_version_cached(taxonomy_text, taxonomy_stamp, lexicon_text, lexicon_stamp):
    del taxonomy_stamp, lexicon_stamp
    digest = hashlib.sha256()
    digest.update(Path(taxonomy_text).read_bytes())
    digest.update(b"\0")
    digest.update(Path(lexicon_text).read_bytes())
    return "kb-" + digest.hexdigest()[:12]


def knowledge_base_version():
    taxonomy = taxonomy_path()
    lexicon = lexicon_path()
    return _knowledge_base_version_cached(str(taxonomy), _stamp(taxonomy), str(lexicon), _stamp(lexicon))


def evidence_version():
    return f"{EXTRACTOR_VERSION}+{knowledge_base_version()}"


def _clause_segment(text, start, end):
    left = 0
    right = len(text)
    for match in CLAUSE_BOUNDARY.finditer(text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    full_clause = text[left:right]
    relative_start = start - left
    relative_end = end - left
    segment_left = 0
    segment_right = len(full_clause)
    markers = []
    for match in CONTRAST.finditer(full_clause):
        markers.append(match.group())
        if match.end() <= relative_start:
            segment_left = match.end()
        elif match.start() >= relative_end:
            segment_right = match.start()
            break
    while segment_left < segment_right and full_clause[segment_left].isspace():
        segment_left += 1
    while segment_right > segment_left and full_clause[segment_right - 1].isspace():
        segment_right -= 1
    context_start = left + segment_left
    context_end = left + segment_right
    return text[context_start:context_end], context_start, context_end, markers


def _matches(pattern, text):
    return bool(pattern and re.search(pattern, text, flags=re.I))


def _statement_type(rule, clause):
    if MARKETING.search(clause):
        return "营销"
    if REQUEST.search(clause):
        return "诉求"
    return rule["statement_type"]


def _resolve_polarity(text, match, rule, clause, context_start, statement_type):
    evidence = match.group()
    prefix = text[max(context_start, match.start() - 6):match.start()]
    negated = bool(NEGATION_PREFIX.search(prefix) or NEGATION_IN_EVIDENCE.search(evidence))
    if statement_type in {"诉求", "营销"}:
        return "不适用", negated, "high", "非体验表达，不赋予消费体验效价", prefix

    strategy = rule["polarity_strategy"]
    if strategy == "none":
        return rule["default_polarity"], negated, "high", "规则声明该表达不使用上下文效价", prefix

    if strategy == "fixed":
        polarity = rule["default_polarity"]
        if negated and polarity == "负面":
            return "中性", True, "medium", "否定词削弱了负向表达，保守标为中性", prefix
        if negated and polarity == "正面":
            return "负面", True, "medium", "否定词反转了明确正向属性", prefix
        return polarity, negated, "high", "规则本身包含明确效价", prefix

    if _matches(rule["neutral_context"], clause):
        polarity, reason, confidence = "中性", "同一子句命中中性上下文", "high"
    elif _matches(rule["negative_context"], clause):
        polarity, reason, confidence = "负面", "同一子句命中负向上下文", "high"
    elif _matches(rule["positive_context"], clause):
        polarity, reason, confidence = "正面", "同一子句命中正向上下文", "high"
    else:
        polarity, reason, confidence = rule["default_polarity"], "上下文不足，保留规则默认效价", "low"
    if negated and polarity in {"正面", "负面"}:
        return "中性", True, "medium", "否定范围可能改变上下文效价，保守标为中性", prefix
    return polarity, negated, confidence, reason, prefix


def extract_text_features(text):
    text = str(text or "")
    kb_version = knowledge_base_version()
    combined_version = f"{EXTRACTOR_VERSION}+{kb_version}"
    candidates = []
    for rule in load_rules():
        for match in re.finditer(rule["pattern"], text, flags=re.I):
            clause, context_start, context_end, markers = _clause_segment(text, match.start(), match.end())
            statement_type = _statement_type(rule, clause)
            polarity, negated, confidence, polarity_basis, prefix = _resolve_polarity(
                text, match, rule, clause, context_start, statement_type)
            candidates.append({
                "feature_version": combined_version,
                "knowledge_base_version": kb_version,
                "extractor_version": EXTRACTOR_VERSION,
                "rule_id": rule["rule_id"],
                "feature": rule["feature"],
                "standard_tag": rule["standard_tag"],
                "evidence": match.group(),
                "start": match.start(),
                "end": match.end(),
                "clause": clause,
                "context_start": context_start,
                "context_end": context_end,
                "statement_type": statement_type,
                "polarity_candidate": polarity,
                "confidence": confidence,
                "negated": negated,
                "negation_context": prefix,
                "contrast_markers": "、".join(markers),
                "basis": rule["basis"],
                "polarity_basis": polarity_basis,
                "decision": "待人工复核" if confidence == "low" or polarity == "待判定" else "规则证据候选",
                "source": "文本规则",
                "priority": rule["priority"],
            })
    selected = []
    for item in sorted(candidates, key=lambda x: (x["start"], -(x["end"] - x["start"]), x["priority"])):
        if any(
            item["feature"] == old["feature"]
            and item["start"] < old["end"]
            and old["start"] < item["end"]
            for old in selected
        ):
            continue
        selected.append(item)
    for item in selected:
        item.pop("priority", None)
    return sorted(selected, key=lambda x: (x["start"], x["feature"]))


def extract_feature_rows(records):
    rows = []
    business = [
        "review_id", "source_row", "month", "platform", "brand", "price", "product_price",
        "sku_id", "sku_name", "comment_id", "category", "text",
    ]
    kb_version = knowledge_base_version()
    combined_version = f"{EXTRACTOR_VERSION}+{kb_version}"
    for record in records:
        base = {key: record.get(key, "") for key in business}
        for item in extract_text_features(record.get("text", "")):
            rows.append({**base, **item})
        category = str(record.get("category", "")).strip()
        if category:
            rows.append({
                **base,
                "feature_version": combined_version,
                "knowledge_base_version": kb_version,
                "extractor_version": EXTRACTOR_VERSION,
                "rule_id": "DATA_CATEGORY",
                "feature": "品类",
                "standard_tag": category,
                "evidence": category,
                "start": None,
                "end": None,
                "clause": "",
                "context_start": None,
                "context_end": None,
                "statement_type": "数据标签",
                "polarity_candidate": "不适用",
                "confidence": "high",
                "negated": False,
                "negation_context": "",
                "contrast_markers": "",
                "basis": "来自业务数据自身的品类字段，不从评论推断",
                "polarity_basis": "业务标签不使用情感效价",
                "decision": "数据标签",
                "source": "数据标签",
            })
    return rows


def rule_catalog():
    return load_rules()


def taxonomy_catalog():
    return load_taxonomy()


def changelog_catalog():
    path = changelog_path()
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def knowledge_base_info():
    taxonomy = load_taxonomy()
    rules = load_rules()
    kb_version = knowledge_base_version()
    return {
        "taxonomy_path": str(taxonomy_path()),
        "lexicon_path": str(lexicon_path()),
        "changelog_path": str(changelog_path()),
        "enabled_features": len(taxonomy),
        "enabled_rules": len(rules),
        "knowledge_base_version": kb_version,
        "extractor_version": EXTRACTOR_VERSION,
        "feature_version": f"{EXTRACTOR_VERSION}+{kb_version}",
    }
