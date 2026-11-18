import io
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
from features import (changelog_catalog, extract_feature_rows, knowledge_base_info, rule_catalog,
    taxonomy_catalog)
from evaluation import build_review_samples, evaluate_gold_standard, quick_review_metrics
from inputs import prepare_reviews


DEFAULT_STOPWORDS = {
    '这个', '那个', '就是', '还是', '已经', '感觉', '觉得', '真的', '比较', '非常', '特别',
    '一个', '没有', '不是', '可以', '但是', '因为', '所以', '而且', '然后', '东西', '产品',
    '收到', '购买', '买了', '一下', '有点', '一样', '这么', '怎么', '什么', '不错', '很好',
}
FONT_CANDIDATES = [
    Path(r'C:\Windows\Fonts\msyh.ttc'),
    Path(r'C:\Windows\Fonts\simhei.ttf'),
    Path('/System/Library/Fonts/PingFang.ttc'),
    Path('/System/Library/Fonts/Hiragino Sans GB.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf'),
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
]


def find_cjk_font():
    configured = os.environ.get('REVIEW_FONT_PATH', '').strip()
    candidates = ([Path(configured).expanduser()] if configured else []) + FONT_CANDIDATES
    return next((str(path) for path in candidates if path.is_file()), None)


def prepare_data(frame, text_col, mapping, dedup_mode='评论ID优先'):
    return prepare_reviews(frame, text_col, mapping, dedup_mode)


def filter_data(data, mapping, months=None, platforms=None, brands=None, price_range=None):
    result = data
    if months and result['_month_dt'].notna().any():
        start, end = pd.Timestamp(months[0]), pd.Timestamp(months[1])
        result = result[result['_month_dt'].between(start, end)]
    for field, selected in [('platform', platforms), ('brand', brands)]:
        col = mapping.get(field)
        if col and selected:
            result = result[result[col].isin(selected)]
    if price_range and result['_price_numeric'].notna().any():
        result = result[result['_price_numeric'].between(*price_range)]
    return result.copy()


@st.cache_data(show_spinner=False)
def token_frequencies(texts, extra_stopwords=None, max_docs=20000):
    stopwords = DEFAULT_STOPWORDS | set(extra_stopwords or [])
    counter = Counter()
    series = pd.Series(texts, dtype=str).dropna()
    if len(series) > max_docs:
        series = series.sample(max_docs, random_state=42)
    for text in series:
        for token in jieba.cut(text, cut_all=False):
            token = token.strip().lower()
            if len(token) < 2 or token in stopwords or not re.search(r'[\u4e00-\u9fffA-Za-z]', token):
                continue
            counter[token] += 1
    return counter


@st.cache_data(show_spinner=False)
def wordcloud_png(frequencies, font_path, width=900, height=420):
    if not frequencies or not font_path:
        return None
    cloud = WordCloud(width=width, height=height, background_color='white', colormap='viridis',
        font_path=font_path, max_words=120, collocations=False).generate_from_frequencies(dict(frequencies))
    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format='PNG')
    return buffer.getvalue()


def price_bands(series):
    bins = [-float('inf'), 10, 20, 30, 50, 80, 120, float('inf')]
    labels = ['≤10', '10–20', '20–30', '30–50', '50–80', '80–120', '>120']
    return pd.cut(series, bins=bins, labels=labels)


def unique_sku_observations(data, mapping):
    priced = data[data['_price_numeric'].notna()].copy()
    sku_col = mapping.get('sku_id')
    if not sku_col or priced.empty:
        return priced.iloc[0:0]
    keys = [c for c in [mapping.get('platform'), mapping.get('brand'), sku_col] if c]
    return (priced.sort_values(['_month_dt', '_source_row'], na_position='first')
        .drop_duplicates(keys, keep='last'))


def feature_records(data, mapping, text_col):
    rows = []
    fields = ['month', 'platform', 'brand', 'price', 'product_price', 'sku_id', 'sku_name', 'comment_id', 'category']
    for _, row in data.iterrows():
        item = {'review_id': str(row['_review_key']), 'source_row': int(row['_source_row']), 'text': str(row[text_col])}
        for field in fields:
            source = mapping.get(field)
            item[field] = str(row[source]) if source else ''
        rows.append(item)
    return rows


@st.cache_data(show_spinner=False)
def cached_feature_rows(records_json, feature_version):
    del feature_version
    return extract_feature_rows(json.loads(records_json))


def render_feature_dashboard(filtered, mapping, text_col):
    records = feature_records(filtered, mapping, text_col)
    kb = knowledge_base_info()
    rows = cached_feature_rows(
        json.dumps(records, ensure_ascii=False, sort_keys=True, default=str), kb['feature_version'])
    feature_df = pd.DataFrame(rows)
    st.markdown('### 特征短语输出')
    st.caption('一级特征固定为少量业务维度，具体问题进入二级标准标签。一条评价可命中多个特征；品类直接读取业务字段。')
    taxonomy = pd.DataFrame(taxonomy_catalog())
    st.caption(f"特征资料库：{kb['enabled_features']}个一级特征，{kb['enabled_rules']}条启用规则；抽取器 {kb['extractor_version']}，知识库 {kb['knowledge_base_version']}。修改CSV后页面会自动重新加载。")
    st.dataframe(taxonomy, use_container_width=True, hide_index=True)
    st.download_button('下载一级特征体系 CSV', taxonomy.to_csv(index=False).encode('utf-8-sig'), 'feature_taxonomy.csv', mime='text/csv')
    with st.expander('查看当前特征标签规则'):
        catalog = pd.DataFrame(rule_catalog())
        st.dataframe(catalog, use_container_width=True, hide_index=True)
        st.download_button('下载特征规则 CSV', catalog.to_csv(index=False).encode('utf-8-sig'), 'feature_rules.csv', mime='text/csv')
        changes = pd.DataFrame(changelog_catalog())
        if not changes.empty:
            st.markdown('#### 知识库变更记录')
            st.dataframe(changes, use_container_width=True, hide_index=True)
            st.download_button('下载知识库变更记录 CSV', changes.to_csv(index=False).encode('utf-8-sig'),
                'knowledge_base_changelog.csv', mime='text/csv')
            if kb['knowledge_base_version'] not in set(changes.get('kb_version', [])):
                st.warning('当前知识库 Hash 尚未写入变更记录；交付前必须补充记录并重新校验。')
    if feature_df.empty:
        st.info('当前筛选范围没有命中特征规则，也没有可用业务品类标签。')
        return rows
    text_features = feature_df[feature_df['source'] == '文本规则']
    f1, f2, f3, f4 = st.columns(4)
    f1.metric('抽取特征短语', f'{len(text_features):,}')
    f2.metric('有文本特征评价', f"{text_features['review_id'].nunique():,}")
    f3.metric('文本规则覆盖率', f"{text_features['review_id'].nunique() / len(filtered):.1%}" if len(filtered) else '—')
    f4.metric('业务品类数', feature_df.loc[feature_df['feature'] == '品类', 'standard_tag'].nunique())

    text_tab, compare_tab, detail_tab = st.tabs(['特征分布', '平台/品牌/价格对比', '证据明细'])
    with text_tab:
        unique_mentions = text_features.drop_duplicates(['review_id', 'feature', 'standard_tag'])
        if text_features.empty:
            st.info('当前仅有业务品类标签，没有文本规则命中。')
        else:
            feature_counts = unique_mentions['feature'].value_counts().rename_axis('特征').reset_index(name='评价数')
            tag_counts = unique_mentions.groupby(['feature', 'standard_tag']).review_id.nunique().reset_index(name='评价数')
            a, b = st.columns(2)
            with a:
                st.plotly_chart(px.bar(feature_counts.sort_values('评价数'), x='评价数', y='特征', orientation='h', title='特征提及评价数'), use_container_width=True)
            with b:
                top_tags = tag_counts.nlargest(20, '评价数').sort_values('评价数')
                st.plotly_chart(px.bar(top_tags, x='评价数', y='standard_tag', color='feature', orientation='h',
                    labels={'standard_tag':'标准标签', 'feature':'特征'}, title='高频标准特征标签'), use_container_width=True)
            sentiment = (text_features.drop_duplicates(['review_id', 'feature', 'polarity_candidate'])
                .groupby(['feature', 'polarity_candidate']).review_id.nunique().reset_index(name='评价数'))
            order = ['正面', '负面', '中性', '待判定', '不适用']
            sentiment['polarity_candidate'] = pd.Categorical(sentiment['polarity_candidate'], order, ordered=True)
            st.plotly_chart(px.bar(sentiment.sort_values('polarity_candidate'), x='feature', y='评价数',
                color='polarity_candidate', barmode='stack',
                category_orders={'polarity_candidate': order},
                color_discrete_map={'正面':'#2ca02c','负面':'#d62728','中性':'#7f7f7f','待判定':'#ffbf00','不适用':'#aec7e8'},
                labels={'feature':'一级特征','polarity_candidate':'情感'}, title='一级特征 × 情感分布'), use_container_width=True)
            pending = text_features[(text_features['polarity_candidate'] == '待判定') | (text_features['confidence'] == 'low')]
            if not pending.empty:
                st.warning(f'有 {pending.review_id.nunique():,} 条评价包含待判定或低置信证据，建议进入人工复核或模型增强候选池。')
                st.download_button('下载待判定/低置信证据 CSV', pending.to_csv(index=False).encode('utf-8-sig'),
                    'pending_evidence.csv', mime='text/csv')
        categories = feature_df[feature_df['feature'] == '品类'].drop_duplicates(['review_id', 'standard_tag'])
        if not categories.empty:
            category_counts = categories.standard_tag.value_counts().rename_axis('业务品类').reset_index(name='评价数')
            st.plotly_chart(px.treemap(category_counts, path=['业务品类'], values='评价数', title='业务品类结构'), use_container_width=True)

    with compare_tab:
        for field, label in [('platform', '平台'), ('brand', '品牌')]:
            if field not in text_features or text_features[field].replace('', pd.NA).dropna().empty:
                continue
            top_groups = text_features[field].value_counts().head(12).index
            subset = text_features[text_features[field].isin(top_groups)].drop_duplicates(['review_id', field, 'feature'])
            denominators = filtered[mapping[field]].value_counts() if mapping.get(field) else pd.Series(dtype=float)
            matrix = subset.groupby([field, 'feature']).review_id.nunique().rename('提及评价数').reset_index()
            matrix['特征提及率'] = matrix.apply(lambda x: x['提及评价数'] / denominators.get(x[field], 1), axis=1)
            pivot = matrix.pivot(index=field, columns='feature', values='特征提及率').fillna(0)
            st.plotly_chart(px.imshow(pivot, text_auto='.1%', aspect='auto', color_continuous_scale='Blues',
                title=f'{label}特征提及率对比'), use_container_width=True)
        priced = text_features.copy()
        priced['_price_numeric'] = pd.to_numeric(priced['price'], errors='coerce')
        priced = priced[priced['_price_numeric'].notna()]
        if not priced.empty:
            priced['价格带'] = price_bands(priced['_price_numeric']).astype(str)
            price_matrix = (priced.drop_duplicates(['review_id', '价格带', 'feature'])
                .groupby(['价格带', 'feature']).review_id.nunique().reset_index(name='提及评价数'))
            st.plotly_chart(px.bar(price_matrix, x='价格带', y='提及评价数', color='feature', barmode='group',
                title='价格带中的特征提及'), use_container_width=True)

    with detail_tab:
        features = ['全部'] + sorted(feature_df.feature.unique())
        selected = st.selectbox('筛选特征', features)
        shown = feature_df if selected == '全部' else feature_df[feature_df.feature == selected]
        st.dataframe(shown.head(10000), use_container_width=True, hide_index=True)
    st.download_button('下载特征短语明细 CSV', feature_df.to_csv(index=False).encode('utf-8-sig'),
        'feature_evidence.csv', mime='text/csv')
    return rows


def render_review_workspace(records, feature_rows):
    st.markdown('### 规则复核与金标准评估')
    st.caption('快速抽查用于发现误报和覆盖缺口；只有完整标注模式会计算正式召回率。复核结果不会自动改写正式词库。')
    if not records:
        st.info('当前没有可复核的评论。')
        return
    mode = st.radio('复核模式', ['快速抽查', '金标准完整标注'], horizontal=True, key='review_workspace_mode')
    c1, c2 = st.columns(2)
    n_hits = c1.number_input('抽取命中评论数', 1, 200, 20, key='review_n_hits')
    n_uncovered = c2.number_input('抽取未覆盖评论数', 1, 200, 20, key='review_n_uncovered')
    samples = build_review_samples(records, feature_rows, n_hits, n_uncovered)
    sample_ids = samples['hit_review_ids'] + samples['uncovered_review_ids']
    signature = hashlib.sha256(json.dumps(sample_ids, ensure_ascii=False).encode()).hexdigest()[:10]

    if mode == '快速抽查':
        st.info('该模式可估算命中证据精确率和未覆盖需求发现率，不能称为正式召回率。')
        hit_frame = pd.DataFrame(samples['quick_hits'])
        uncovered_frame = pd.DataFrame(samples['quick_uncovered'])
        if not hit_frame.empty:
            st.markdown('#### 命中证据复核')
            edited_hits = st.data_editor(
                hit_frame,
                key='quick_hits_' + signature,
                hide_index=True,
                use_container_width=True,
                column_config={'review_result': st.column_config.SelectboxColumn(
                    '人工判定', options=['未复核', '正确', '错误'], required=True)},
                disabled=['review_id', 'text', 'rule_id', 'feature', 'standard_tag', 'evidence', 'polarity', 'statement_type'],
            )
        else:
            edited_hits = hit_frame
            st.info('没有规则命中样本。')
        if not uncovered_frame.empty:
            st.markdown('#### 未覆盖评论复核')
            edited_uncovered = st.data_editor(
                uncovered_frame,
                key='quick_uncovered_' + signature,
                hide_index=True,
                use_container_width=True,
                column_config={'coverage_review': st.column_config.SelectboxColumn(
                    '人工判定', options=['未复核', '有遗漏', '无遗漏'], required=True)},
                disabled=['review_id', 'text'],
            )
        else:
            edited_uncovered = uncovered_frame
            st.info('没有未覆盖样本。')
        metrics = quick_review_metrics(edited_hits.to_dict('records'), edited_uncovered.to_dict('records'))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('已复核证据', metrics['reviewed_evidence'])
        m2.metric('抽样精确率', f"{metrics['estimated_precision']:.1%}" if metrics['estimated_precision'] is not None else '—')
        m3.metric('已复核未覆盖评论', metrics['reviewed_uncovered_reviews'])
        m4.metric('未覆盖需求发现率', f"{metrics['uncovered_demand_discovery_rate']:.1%}" if metrics['uncovered_demand_discovery_rate'] is not None else '—')
        feedback = pd.concat([
            edited_hits.assign(sample_type='命中证据'),
            edited_uncovered.assign(sample_type='未覆盖评论'),
        ], ignore_index=True, sort=False)
        st.download_button('下载复核反馈候选 CSV', feedback.to_csv(index=False).encode('utf-8-sig'),
            'rule_review_feedback.csv', mime='text/csv')
        st.caption('误报和遗漏只进入候选反馈；确认规则修改并通过回归测试后，才能写入正式词库。')
        return

    st.warning('正式召回率要求：样本中每条评论的全部真实特征、标签、证据跨度、情感和语句类型都必须标完；无特征评论必须勾选 gold_no_feature。')
    template = pd.DataFrame(samples['gold_template'])
    st.download_button('下载金标准标注模板 CSV', template.to_csv(index=False).encode('utf-8-sig'),
        'gold_annotation_template.csv', mime='text/csv')
    uploaded = st.file_uploader('可选：上传已填写的金标准 CSV', type=['csv'], key='gold_upload_' + signature)
    if uploaded is not None:
        try:
            editable = pd.read_csv(uploaded, encoding='utf-8-sig', dtype=str).fillna('')
        except Exception as exc:
            st.error('金标准文件读取失败：' + str(exc))
            editable = template
    else:
        editable = template
    edited_gold = st.data_editor(
        editable,
        key='gold_editor_' + signature + ('_' + uploaded.name if uploaded is not None else ''),
        num_rows='dynamic',
        hide_index=True,
        use_container_width=True,
        column_config={
            'annotation_complete': st.column_config.CheckboxColumn('本行标注完成'),
            'gold_no_feature': st.column_config.CheckboxColumn('确认无特征'),
            'gold_polarity': st.column_config.SelectboxColumn('人工情感',
                options=['', '正面', '负面', '中性', '待判定', '不适用']),
            'gold_statement_type': st.column_config.SelectboxColumn('人工语句类型',
                options=['', '体验', '诉求', '场景', '关注', '描述', '营销', '未知']),
            'adjudication_status': st.column_config.SelectboxColumn('仲裁状态',
                options=['未仲裁', '一致', '已仲裁']),
        },
    )
    metric_key = 'gold_metrics_' + signature
    if st.button('计算正式精确率 / 召回率 / F1', key='evaluate_gold_' + signature):
        try:
            st.session_state[metric_key] = evaluate_gold_standard(feature_rows, edited_gold.to_dict('records'), sample_ids)
        except Exception as exc:
            st.error(str(exc))
            st.session_state.pop(metric_key, None)
    metrics = st.session_state.get(metric_key)
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('证据精确率', f"{metrics['precision_relaxed']:.1%}")
        m2.metric('证据召回率', f"{metrics['recall_relaxed']:.1%}")
        m3.metric('证据 F1', f"{metrics['f1_relaxed']:.1%}")
        polarity = metrics['polarity_accuracy_given_matched_feature']
        m4.metric('匹配特征后的情感准确率', f'{polarity:.1%}' if polarity is not None else '—')
        st.caption(f"评估单位：证据片段；样本 {metrics['sampled_reviews']} 条，系统证据 {metrics['predicted_evidence']} 条，人工证据 {metrics['gold_evidence']} 条。")
        errors = pd.concat([
            pd.DataFrame(metrics['false_positives']).assign(error_type='误报'),
            pd.DataFrame(metrics['false_negatives']).assign(error_type='漏报'),
        ], ignore_index=True, sort=False)
        if not errors.empty:
            st.dataframe(errors, use_container_width=True, hide_index=True)
            st.download_button('下载误报与漏报 CSV', errors.to_csv(index=False).encode('utf-8-sig'),
                'gold_evaluation_errors.csv', mime='text/csv')
        st.download_button('下载评估指标 JSON', json.dumps({k: v for k, v in metrics.items()
            if k not in ['false_positives', 'false_negatives']}, ensure_ascii=False, indent=2),
            'gold_evaluation_metrics.json', mime='application/json')


def render_dashboard(data, mapping, text_col):
    st.subheader('类目经营看板')
    st.caption('所有图表由同一组筛选器联动。词频最多抽取20,000条筛选后评论，图表不调用模型。')
    f1, f2, f3, f4 = st.columns(4)
    month_range = None
    if data['_month_dt'].notna().any():
        lo, hi = data['_month_dt'].min().date(), data['_month_dt'].max().date()
        month_range = f1.date_input('评价月份范围', value=(lo, hi), min_value=lo, max_value=hi)
        if not isinstance(month_range, (list, tuple)) or len(month_range) != 2:
            month_range = (lo, hi)
    platform_col, brand_col = mapping.get('platform'), mapping.get('brand')
    platforms = f2.multiselect('平台', sorted(data[platform_col].dropna().unique())) if platform_col else []
    brands = f3.multiselect('品牌', sorted(data[brand_col].dropna().unique()), max_selections=20) if brand_col else []
    price_range = None
    valid_prices = data['_price_numeric'].dropna()
    if not valid_prices.empty:
        pmin, pmax = float(valid_prices.min()), float(valid_prices.max())
        price_range = f4.slider('SKU价格范围', pmin, pmax, (pmin, pmax)) if pmin < pmax else (pmin, pmax)
    filtered = filter_data(data, mapping, month_range, platforms, brands, price_range)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('筛选后评价', f'{len(filtered):,}')
    k2.metric('平台数', filtered[platform_col].nunique() if platform_col else '—')
    k3.metric('品牌数', filtered[brand_col].nunique() if brand_col else '—')
    k4.metric('SKU数', filtered[mapping['sku_id']].nunique() if mapping.get('sku_id') else '—')
    k5.metric('价格中位数', f"¥{filtered['_price_numeric'].median():.2f}" if filtered['_price_numeric'].notna().any() else '—')

    feature_rows = render_feature_dashboard(filtered, mapping, text_col)
    with st.expander('打开规则复核与金标准评估', expanded=False):
        render_review_workspace(feature_records(filtered, mapping, text_col), feature_rows)

    custom = st.text_input('词云排除词（逗号分隔）', value='物流,快递,客服,包装', help='修改后会同步更新总体及分组词云。')
    stop = {x.strip() for x in re.split(r'[,，]', custom) if x.strip()}
    frequencies = token_frequencies(filtered[text_col], stop)
    font_path = find_cjk_font()
    if not font_path:
        st.warning('未找到可用中文字体，已跳过词云。可通过 REVIEW_FONT_PATH 指定中文 TTF/OTF/TTC 字体。')
    cloud = wordcloud_png(tuple(frequencies.most_common(300)), font_path) if font_path else None
    left, right = st.columns([2, 1])
    with left:
        st.markdown('#### 类目整体词云')
        if cloud:
            st.image(cloud, use_column_width=True)
        else:
            st.info('筛选范围内没有可生成词云的词。')
    with right:
        st.markdown('#### 高频词')
        top = pd.DataFrame(frequencies.most_common(20), columns=['词语', '次数'])
        st.dataframe(top, hide_index=True, use_container_width=True)

    if platform_col:
        st.markdown('#### 平台词云对比')
        available = filtered[platform_col].value_counts().index.tolist()
        selected = st.multiselect('选择对比平台', available, default=available[:min(3, len(available))], max_selections=4)
        cols = st.columns(max(1, len(selected)))
        for container, platform in zip(cols, selected):
            freq = token_frequencies(filtered.loc[filtered[platform_col] == platform, text_col], stop)
            image = wordcloud_png(tuple(freq.most_common(250)), font_path, 600, 360) if font_path else None
            with container:
                st.caption(f'{platform} · {int((filtered[platform_col] == platform).sum()):,}条')
                if image:
                    st.image(image, use_column_width=True)

    if brand_col:
        with st.expander('品牌词云对比', expanded=False):
            available_brands = filtered[brand_col].value_counts().index.tolist()
            selected_brands = st.multiselect('选择对比品牌', available_brands,
                default=available_brands[:min(3, len(available_brands))], max_selections=4)
            brand_cols = st.columns(max(1, len(selected_brands)))
            for container, brand in zip(brand_cols, selected_brands):
                freq = token_frequencies(filtered.loc[filtered[brand_col] == brand, text_col], stop)
                image = wordcloud_png(tuple(freq.most_common(250)), font_path, 600, 360) if font_path else None
                with container:
                    st.caption(f'{brand} · {int((filtered[brand_col] == brand).sum()):,}条')
                    if image:
                        st.image(image, use_column_width=True)

    chart1, chart2 = st.columns(2)
    with chart1:
        if brand_col:
            brand_stats = (filtered.groupby(brand_col).agg(评价数=(text_col, 'size'),
                SKU数=(mapping['sku_id'], 'nunique') if mapping.get('sku_id') else (text_col, 'size')).reset_index())
            sku_level = unique_sku_observations(filtered, mapping)
            if not sku_level.empty:
                brand_stats = brand_stats.merge(
                    sku_level.groupby(brand_col)['_price_numeric'].median().rename('价格中位数'),
                    on=brand_col, how='left')
            else:
                brand_stats['价格中位数'] = float('nan')
            brand_stats = brand_stats.nlargest(15, '评价数')
            st.plotly_chart(px.bar(brand_stats.sort_values('评价数'), x='评价数', y=brand_col,
                orientation='h', color='价格中位数', title='品牌评价规模与SKU价格'), use_container_width=True)
    with chart2:
        if platform_col:
            platform_stats = filtered[platform_col].value_counts().rename_axis('平台').reset_index(name='评价数')
            st.plotly_chart(px.bar(platform_stats, x='平台', y='评价数', title='平台评价结构'), use_container_width=True)

    c1, c2 = st.columns(2)
    sku_level = unique_sku_observations(filtered, mapping)
    if not sku_level.empty:
        sku_level['价格带'] = price_bands(sku_level['_price_numeric']).astype(str)
        with c1:
            band = sku_level['价格带'].value_counts().rename_axis('价格带').reset_index(name='SKU数')
            st.plotly_chart(px.bar(band, x='价格带', y='SKU数', title='SKU价格带结构'), use_container_width=True)
        with c2:
            color = platform_col if platform_col else brand_col
            st.plotly_chart(px.box(sku_level, x=color, y='_price_numeric', points=False,
                labels={'_price_numeric':'SKU价格'}, title='平台SKU价格分布' if platform_col else '品牌SKU价格分布'), use_container_width=True)

    if filtered['_month_dt'].notna().any():
        trend = filtered.assign(月份=filtered['_month_dt'].dt.to_period('M').astype(str)).groupby('月份').size().reset_index(name='评价数')
        if len(trend) > 1:
            st.plotly_chart(px.line(trend, x='月份', y='评价数', markers=True, title='评价量趋势（筛选用途，不解释为需求变化）'), use_container_width=True)
        else:
            st.info('当前筛选范围只有一个月份，暂不绘制趋势线。月份仅用于筛选。')

    with st.expander('评价明细下钻与导出'):
        columns = [c for c in [mapping.get('month'), platform_col, brand_col, mapping.get('sku_id'), mapping.get('sku_name'), mapping.get('price'), text_col] if c]
        st.dataframe(filtered[columns].head(5000), use_container_width=True, hide_index=True)
        st.download_button('下载当前筛选评价 CSV', filtered[columns].to_csv(index=False).encode('utf-8-sig'), 'filtered_reviews.csv', mime='text/csv')
        st.download_button('下载当前词频 CSV', pd.DataFrame(frequencies.most_common(), columns=['词语','次数']).to_csv(index=False).encode('utf-8-sig'), 'token_frequencies.csv', mime='text/csv')
    return filtered, feature_rows
