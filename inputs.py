import os
from pathlib import Path

import pandas as pd

TEST_FILE = Path(os.environ.get('REVIEW_AGENT_TEST_FILE', '__not_configured_test__.xlsx')).expanduser()
BUSINESS_FILE = Path(os.environ.get('REVIEW_AGENT_BUSINESS_FILE', '__not_configured_business__.xlsx')).expanduser()
ALIASES = {'month': ['月份', 'month'], 'platform': ['平台', 'platform'],
    'brand': ['品牌', 'brand'], 'price': ['SKU价格', '价格', 'price'],
    'product_price': ['价格'], 'sku_id': ['SKUID', 'SKU ID', 'sku_id'],
    'sku_name': ['SKU名称', 'sku_name'], 'comment_id': ['评论ID', 'comment_id']}
DEFAULT_CATEGORY_ALIASES = ['四级类目名称', '三级类目名称', '二级类目名称', '品类', 'category']
ALIASES['category'] = [item.strip() for item in os.environ.get(
    'REVIEW_CATEGORY_ALIASES', ','.join(DEFAULT_CATEGORY_ALIASES)).split(',') if item.strip()]


def infer_mapping(frame):
    return {field: next((c for c in candidates if c in frame.columns), '') for field, candidates in ALIASES.items()}


def read_source(source, encoding='utf-8-sig', nrows=10000):
    name = str(getattr(source, 'name', source)).lower()
    if name.endswith('.xlsx'):
        frame = pd.read_excel(source, nrows=nrows, dtype=str, engine='openpyxl')
    else:
        frame = pd.read_csv(source, encoding=encoding, nrows=nrows, dtype=str)
    return frame.fillna('')


def prepare_reviews(frame, text_col, mapping, dedup_mode='评论ID优先'):
    """Normalize and deduplicate once; all downstream consumers use this frame."""
    data = frame.copy()
    data['_source_row'] = range(2, len(data) + 2)
    data['_normalized_text'] = (data[text_col].fillna('').astype(str)
        .str.replace(r'\s+', ' ', regex=True).str.strip())
    data = data[data['_normalized_text'].ne('')].copy()
    month_col = mapping.get('month')
    data['_month_dt'] = pd.to_datetime(data[month_col], errors='coerce') if month_col else pd.NaT
    price_col = mapping.get('price')
    data['_price_numeric'] = pd.to_numeric(data[price_col], errors='coerce') if price_col else float('nan')
    platform = data[mapping['platform']].fillna('').astype(str) if mapping.get('platform') else pd.Series('', index=data.index)
    sku = data[mapping['sku_id']].fillna('').astype(str) if mapping.get('sku_id') else pd.Series('', index=data.index)
    comment = data[mapping['comment_id']].fillna('').astype(str) if mapping.get('comment_id') else pd.Series('', index=data.index)
    if dedup_mode == '全局相同文本':
        key = data['_normalized_text']
    elif dedup_mode == '同SKU相同文本':
        key = platform + '|' + sku + '|' + data['_normalized_text']
    else:
        fallback = platform + '|' + sku + '|text|' + data['_normalized_text']
        key = (platform + '|' + sku + '|id|' + comment).where(comment.str.strip().ne(''), fallback)
    data['_review_key'] = key
    data['_row_order'] = range(len(data))
    data = data.sort_values(['_month_dt', '_row_order'], na_position='first')
    data = data.drop_duplicates('_review_key', keep='last').sort_values('_source_row')
    nonempty_count = int(frame[text_col].fillna('').astype(str).str.strip().ne('').sum())
    quality = {
        '原始行数': len(frame),
        '非空评论行数': nonempty_count,
        '去重后评论数': len(data),
        '去除重复行数': int(nonempty_count - len(data)),
        '月份可解析数': int(data['_month_dt'].notna().sum()),
        '价格可解析数': int(data['_price_numeric'].notna().sum()),
    }
    return data, quality


def make_records(frame, column, limit, mapping=None):
    mapping = infer_mapping(frame) if mapping is None else mapping
    if '_review_key' in frame.columns:
        clean = frame[frame[column].fillna('').astype(str).str.strip().ne('')].copy()
    else:
        clean, _ = prepare_reviews(frame, column, mapping, '评论ID优先')
    chosen = clean.sample(n=min(int(limit), len(clean)), random_state=42).sort_index()
    records = []
    for i, row in chosen.iterrows():
        source_row = int(row.get('_source_row', int(i) + 2))
        stable_id = str(row.get('_review_key', i))
        record = {'review_id': stable_id, 'source_row': source_row,
                  'text': str(row[column]), 'source_column': column}
        if column == 'original_text':
            record['original_text'] = str(row[column])
        if 'post_id' in frame:
            record['post_id'] = str(row['post_id'])
        for field, source_col in mapping.items():
            if source_col:
                record[field] = str(row[source_col])
        records.append(record)
    return records
