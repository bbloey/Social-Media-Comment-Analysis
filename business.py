import pandas as pd


def segment_statistics(reviews, opinions, dimension):
    """Same successful-experience denominator as overall, separately per segment."""
    valid = [r for r in reviews if r.get('kind') == '消费体验' and str(r.get(dimension, '')).strip()]
    lookup = {r['review_id']: str(r[dimension]) for r in valid}
    denominators = pd.Series(list(lookup.values()), dtype=str).value_counts().to_dict()
    groups = {}
    for opinion in opinions:
        rid = opinion['review_id']
        if rid not in lookup:
            continue
        key = (lookup[rid], opinion['category'])
        group = groups.setdefault(key, {'mentions': set(), 'negative': set()})
        group['mentions'].add(rid)
        if opinion['sentiment'] in ['负面', '混合']:
            group['negative'].add(rid)
    return pd.DataFrame([{'分组': seg, '需求分类': cat, '有效体验评论数': denominators[seg],
        '提及评论数': len(g['mentions']), '含负面评论数': len(g['negative']),
        '提及率': len(g['mentions']) / denominators[seg],
        '含负面率': len(g['negative']) / len(g['mentions'])} for (seg, cat), g in groups.items()])


def sku_prices(records):
    frame = pd.DataFrame(records)
    if 'price' not in frame:
        return pd.DataFrame()
    frame['price_numeric'] = pd.to_numeric(frame['price'], errors='coerce')
    frame = frame[frame.price_numeric.gt(0)]
    keys = [k for k in ['month', 'platform', 'brand', 'sku_id'] if k in frame]
    if not keys:
        return pd.DataFrame()
    return frame.groupby(keys, dropna=False)['price_numeric'].agg(最低观测价='min', 最高观测价='max', 中位观测价='median').reset_index()
