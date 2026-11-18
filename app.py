import io
import json
import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import Model, extract, discover, statistics, opportunities
from inputs import TEST_FILE, BUSINESS_FILE, read_source, make_records, infer_mapping
from business import segment_statistics, sku_prices
from research import analyze_signals, hierarchy, synthesize, save_checkpoint, html_report
from audit import audit_text, audit_csv_rows
from dashboard import prepare_data, render_dashboard


@st.cache_data(show_spinner=False)
def load_local_source(path, encoding, nrows):
    return read_source(Path(path), encoding, nrows)


@st.cache_data(show_spinner=False)
def load_uploaded_source(content, filename, encoding, nrows):
    stream = io.BytesIO(content)
    stream.name = filename
    return read_source(stream, encoding, nrows)

st.set_page_config(page_title='消费者需求研究 Agent', page_icon='🔎', layout='wide')
st.title('消费者需求研究 Agent')
st.caption('业务数据预处理 → 交互式类目看板 → 需求情感分析 → 开品假设。看板不依赖模型。')
goal = st.text_input('研究目标 / 待验证的开品方向', placeholder='例如：办公场景下的小份低糖零食，消费者有哪些未满足需求？')
goal_key = goal.strip()
if st.session_state.get('goal_key') != goal_key:
    for field in ['signals', 'research_report']:
        st.session_state.pop(field, None)
    st.session_state['goal_key'] = goal_key
with st.sidebar:
    st.header('模型连接')
    base = st.text_input('API Base URL', os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    name = st.text_input('模型名称', os.getenv('OPENAI_MODEL', 'gpt-5.6-sol'))
    key = st.text_input('API Key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
    mode_options = ['OpenAI Responses API（推荐）', '兼容 Chat Completions']
    default_mode = 1 if os.getenv('OPENAI_API_MODE', 'responses') == 'chat_completions' else 0
    mode_label = st.selectbox('API 协议', mode_options, index=default_mode)
    api_mode = 'responses' if mode_label.startswith('OpenAI') else 'chat_completions'
    effort_options = ['none', 'low', 'medium', 'high', 'xhigh', 'max']
    default_effort = os.getenv('OPENAI_REASONING_EFFORT', 'low')
    if default_effort not in effort_options:
        default_effort = 'low'
    reasoning_effort = st.selectbox(
        '推理强度', effort_options, index=effort_options.index(default_effort),
        disabled=api_mode != 'responses', help='评论抽取建议 low；复杂综合报告可提高，但延迟和费用会增加。')
    build_model = lambda: Model(key, base, name, api_mode=api_mode, reasoning_effort=reasoning_effort)
    st.caption('默认使用 GPT-5.6 Sol + OpenAI Responses API。API Key 仅保存在当前页面会话或环境变量中，不写入项目文件。')
    st.caption('点击分析会向所配置接口发送所选评论；本地缓存保留分析结果和原文证据。')
    with st.expander('如何配置 API？', expanded=not bool(key)):
        st.markdown('1. 在 OpenAI API 平台创建 API Key，并确保项目具有 GPT-5.6 Sol 使用权限。\n2. 将密钥粘贴到上面的密码框。\n3. 保持 Base URL 为 `https://api.openai.com/v1`、模型为 `gpt-5.6-sol`、协议为 Responses API。\n4. 点击连接测试，通过后再运行分析。')
        st.markdown('[OpenAI API 密钥管理](https://platform.openai.com/api-keys) · [GPT-5.6 Sol 官方文档](https://developers.openai.com/api/docs/models/gpt-5.6-sol)')
        st.caption('ChatGPT 网页订阅与 API 额度分别计费。密钥请直接填在本地页面，不需要发到聊天中。')
    if st.button('测试模型连接', disabled=not key):
        try:
            with st.spinner('验证接口和 JSON 输出能力…'):
                client = build_model()
                check = client.ask('只返回 JSON 对象 {"ok":true}。', {'task': 'connection_test'})
                if check.get('ok') is not True:
                    raise ValueError('接口可访问，但JSON测试未通过')
            st.success(f'连接成功：{name} · {mode_label}')
            if client.last_metadata.get('service_tier'):
                st.caption('服务层级：' + str(client.last_metadata['service_tier']))
        except Exception as exc:
            st.error(str(exc))

with st.expander('使用步骤', expanded='extraction' not in st.session_state):
    st.write('① 读取与映射字段 → ② 预处理和筛选 → ③ 查看类目经营看板 → ④ 按筛选结果抽取需求 → ⑤ 生成开品假设。')
    st.caption('数据看板和离线判定无需模型；模型只用于后续观点、需求归并和研究报告。')

sample = 'post_id,post_content\n1,味道不错，但太甜了，希望有低糖款。\n2,一大袋拆开吃不完，第二天就不脆了，上班带着不方便。\n3,小袋装很方便，每天上班带一包。\n4,新品上市快来抢购！\n5,封口不好撕，每次都要找剪刀。\n6,希望出椒盐味，现在这个辣味太重。\n'
uploaded = st.file_uploader('上传 CSV / Excel 文件', type=['csv', 'xlsx'])
mode = st.radio('数据来源', ['预配置业务数据', '预配置 original_text 测试文件', '上传文件', '示例评论'], index=0 if BUSINESS_FILE.exists() else 1 if TEST_FILE.exists() else 3)
demo = mode == '示例评论'
encoding = st.selectbox('文件编码', ['utf-8-sig', 'gb18030'])
read_choice = st.selectbox('读取数据行数', ['10,000（快速预览）', '50,000', '100,000', '全部数据（较慢）'])
read_nrows = {'10,000（快速预览）':10000, '50,000':50000, '100,000':100000, '全部数据（较慢）':None}[read_choice]
limit = st.number_input('AI分析最多处理条数', 1, 2000, 30, help='数据看板使用筛选后的全部已读取数据；此限制只影响模型调用。')
if mode != '上传文件' or uploaded is not None:
    try:
        if demo:
            source = read_source(io.StringIO(sample), encoding, read_nrows)
        elif mode == '上传文件':
            source = load_uploaded_source(uploaded.getvalue(), uploaded.name, encoding, read_nrows)
        else:
            selected_path = BUSINESS_FILE if mode == '预配置业务数据' else TEST_FILE
            with st.spinner('首次读取可能较慢，后续筛选会使用缓存…'):
                source = load_local_source(str(selected_path), encoding, read_nrows)
        if mode == '预配置 original_text 测试文件' and 'original_text' not in source:
            raise ValueError('预配置测试文件必须包含 original_text 列')
    except Exception as exc:
        st.error('读取失败，请检查文件路径、编码或表格格式：' + str(exc))
        st.stop()
    st.caption(f'当前读取 {len(source):,} 行。类目ID及历史指标、观点、极性不参与新模型判断。')
    preferred = next((c for c in ['评论内容', 'original_text', 'post_content'] if c in source), source.columns[0])
    col = st.selectbox('评论文本列', source.columns, index=list(source.columns).index(preferred), disabled=mode == '预配置 original_text 测试文件')
    st.dataframe(source.head(8), use_container_width=True)
    mapping = infer_mapping(source)
    with st.expander('业务字段映射 · 类目 ID 无需填写'):
        st.caption('评论必填；其余字段可缺失。价格仅表示表内观测值，并不自动代表成交价或支付意愿。')
        labels = {'month':'月份', 'platform':'平台', 'brand':'品牌', 'price':'分析价格（默认SKU价格）', 'product_price':'商品价格', 'sku_id':'SKU ID', 'sku_name':'SKU名称', 'comment_id':'评论ID', 'category':'业务品类标签'}
        for field, label in labels.items():
            options = ['不提供'] + list(source.columns)
            selected = st.selectbox(label, options, index=options.index(mapping[field]) if mapping[field] else 0, key='map_' + field)
            mapping[field] = '' if selected == '不提供' else selected
        missing = [labels[k] for k in ['month', 'platform', 'brand', 'price'] if not mapping[k]]
        if missing:
            st.info('缺少' + '、'.join(missing) + '：对应分组或价格分析不可用，仍可分析评论。')
    st.subheader('数据预处理')
    dedup_mode = st.selectbox('去重规则', ['评论ID优先', '同SKU相同文本', '全局相同文本'], help='推荐评论ID优先；同一身份保留月份最新的一条。全局相同文本会删掉不同SKU中的通用短评。')
    prepared, quality = prepare_data(source, col, mapping, dedup_mode)
    quality_cols = st.columns(len(quality))
    for container, (label, value) in zip(quality_cols, quality.items()):
        container.metric(label, f'{value:,}')
    st.caption('处理顺序：清理空白 → 标准化文本 → 解析月份与价格 → 构建评论身份 → 按月份保留最新记录。月份缺失时按源文件最后出现的记录保留。')
    filtered_source, feature_rows = render_dashboard(prepared, mapping, col)
    all_records = make_records(filtered_source, col, len(filtered_source), mapping)
    text_rows = [row for row in feature_rows if row.get('source') == '文本规则']
    hit_ids = {str(row['review_id']) for row in text_rows}
    pending_ids = {str(row['review_id']) for row in text_rows
        if row.get('polarity_candidate') == '待判定' or row.get('confidence') == 'low'}
    all_ids = {str(row['review_id']) for row in all_records}
    uncovered_ids = all_ids - hit_ids
    candidate_ids = uncovered_ids | pending_ids
    uncovered_rate = len(uncovered_ids) / len(all_records) if all_records else 0
    pending_rate = len(pending_ids) / len(hit_ids) if hit_ids else 0
    st.caption(f'规则覆盖提示：未覆盖 {len(uncovered_ids):,} 条（{uncovered_rate:.1%}）；命中评论中待判定/低置信 {len(pending_ids):,} 条（{pending_rate:.1%}）。')
    if len(all_records) >= 100 and (uncovered_rate >= float(os.getenv('REVIEW_UNCOVERED_ALERT', '0.35'))
            or pending_rate >= float(os.getenv('REVIEW_PENDING_ALERT', '0.25'))):
        st.warning('当前规则覆盖缺口或待判定比例较高，建议优先对候选池运行模型增强，并将结果进入人工复核。')
    elif len(all_records) < 100:
        st.info('当前筛选样本少于100条，覆盖比例波动较大；建议结合人工抽查判断是否运行模型增强。')
    model_scope = st.selectbox('AI模型增强范围', ['推荐：未覆盖 + 待判定/低置信', '当前筛选全部评论'])
    model_pool = ([row for row in all_records if str(row['review_id']) in candidate_ids]
        if model_scope.startswith('推荐') else all_records)
    if len(model_pool) > int(limit):
        selected_indexes = sorted(random.Random(42).sample(range(len(model_pool)), int(limit)))
        records = [model_pool[i] for i in selected_indexes]
    else:
        records = model_pool
    st.write(f'看板筛选后共 {len(filtered_source):,} 条评价；模型候选池 {len(model_pool):,} 条，其中抽取 {len(records)} 条进入AI分析，保留SKU、业务字段及原始行号。')
    prices = sku_prices(records)
    if not prices.empty:
        with st.expander('SKU观测价格概览'):
            st.dataframe(prices, use_container_width=True)
            st.caption('按SKU及月份/平台汇总，避免评论多的SKU被重复加权。尚未统一克重、件数和促销口径，不能直接作为开品定价。')
            st.download_button('下载SKU价格 CSV', prices.to_csv(index=False).encode('utf-8-sig'), 'sku_prices.csv')
    input_signature = json.dumps(records, ensure_ascii=False, sort_keys=True)
    if st.session_state.get('input_signature') != input_signature:
        for field in ['extraction', 'coded', 'opportunities', 'report_signature', 'signals', 'research_report', 'taxonomy', 'audit_rows']:
            st.session_state.pop(field, None)
        st.session_state['input_signature'] = input_signature
    st.session_state['current_records'] = records
    st.subheader('独立判定依据（离线可用）')
    st.caption('展示可重复执行的规则轨迹，不是模型内部思维。未命中只能判定为规则未覆盖；规则候选不自动进入消费需求统计。')
    if st.button('查看原文判定依据 · 无需 API'):
        st.session_state['audit_rows'] = [{**r, **audit_text(r['text'])} for r in records]
    if 'audit_rows' in st.session_state:
        audit_rows = st.session_state['audit_rows']
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != 'traces'} for r in audit_rows]), use_container_width=True)
        trace_rows = [{'review_id': r['review_id'], **t} for r in audit_rows for t in r['traces']]
        if trace_rows:
            st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)
        st.download_button('下载判定依据 CSV', pd.DataFrame(audit_csv_rows(audit_rows)).to_csv(index=False).encode('utf-8-sig'), 'decision_audit.csv', mime='text/csv')
    if st.button('1. 抽取观点', type='primary', disabled=not key or not records):
        st.session_state.pop('coded', None)
        st.session_state.pop('opportunities', None)
        bar = st.progress(0.0)
        with st.spinner('逐条抽取并核验原文证据…'):
            reviews, opinions, errors = extract(build_model(), records, Path(__file__).parent / 'outputs' / 'cache', lambda n, total: bar.progress(n / total))
        st.session_state['extraction'] = {'reviews': reviews, 'opinions': opinions, 'errors': errors, 'selected': len(records)}

if 'extraction' in st.session_state:
    data = st.session_state['extraction']
    reviews, opinions = data['reviews'], data['opinions']
    st.subheader('抽取结果')
    st.write(f"选中 {data['selected']} 条 · 成功 {len(reviews)} 条 · 观点 {len(opinions)} 条 · 失败 {len(data['errors'])} 条")
    m1, m2, m3 = st.columns(3)
    m1.metric('成功处理文本', len(reviews))
    m2.metric('消费体验文本', sum(x['kind'] == '消费体验' for x in reviews))
    m3.metric('抽取观点', len(opinions))
    if reviews:
        st.bar_chart(pd.DataFrame(reviews)['kind'].value_counts().rename('文本数'))
    st.dataframe(pd.DataFrame(reviews), use_container_width=True)
    if data['errors']:
        st.warning('失败条目未纳入统计；再次抽取会复用成功缓存并重试失败条目。')
        st.dataframe(data['errors'])
        st.info('模型失败条目的 rule_audit 仍包含离线判定依据；这些候选未混入模型观点和统计分母。')
    st.dataframe(pd.DataFrame(opinions), use_container_width=True)
    if st.button('2. 发现并归并需求', disabled=not opinions or not key):
        try:
            with st.spinner('根据观点发现分类，可复用已有类别并新增类别…'):
                st.session_state['coded'] = discover(build_model(), opinions)
                st.session_state.pop('opportunities', None)
        except Exception as exc:
            st.error(str(exc))

coded = st.session_state.get('coded', [])
if coded:
    st.subheader('校订分类与统计')
    st.caption('可直接修改 category 列，将同义类别合并。其他抽取错误请在导出的明细中标注复核。')
    edited = st.data_editor(pd.DataFrame(coded), disabled=[k for k in coded[0] if k != 'category'], hide_index=True, use_container_width=True)
    edited['category'] = edited['category'].fillna('').str.strip().replace('', '待复核')
    coded = edited.to_dict('records')
    st.session_state['coded'] = coded
    denominator = sum(x['kind'] == '消费体验' for x in st.session_state['extraction']['reviews'])
    stats = statistics(coded, denominator)
    st.caption(f'分母：成功判定为消费体验的 {denominator} 条去重文本。含负面包括混合评价；多标签提及率之和可以超过100%。提及率不代表购买重要性。')
    st.dataframe(stats, use_container_width=True)
    st.bar_chart(stats.set_index('需求分类')['提及评论数'])
    with st.expander('业务对比：平台 / 品牌 / 月份'):
        dimensions = [k for k in ['platform', 'brand', 'month'] if any(r.get(k) for r in st.session_state['extraction']['reviews'])]
        if dimensions:
            dim = st.selectbox('对比维度', dimensions, format_func=lambda x: {'platform':'平台','brand':'品牌','month':'月份'}[x])
            segmented = segment_statistics(st.session_state['extraction']['reviews'], coded, dim)
            st.dataframe(segmented, use_container_width=True)
            st.caption('各分组使用各自有效体验评论数作分母；样本量、SKU结构和价格差异可能影响结果，不能直接解释为平台或品牌造成的差异。')
            st.download_button('下载分组对比 CSV', segmented.to_csv(index=False).encode('utf-8-sig'), 'segment_statistics.csv')
        else:
            st.info('没有可用业务分组字段。')
    signature = json.dumps(coded, ensure_ascii=False, sort_keys=True)
    if st.button('3. 生成开品假设', disabled=not key):
        try:
            with st.spinner('结合统计和原文证据生成待验证方案…'):
                st.session_state['opportunities'] = opportunities(build_model(), coded, stats)
                st.session_state['report_signature'] = signature
        except Exception as exc:
            st.error(str(exc))
    items = st.session_state.get('opportunities', []) if st.session_state.get('report_signature') == signature else []
    report = ['# 消费者需求研究报告', f'消费体验统计分母：{denominator}。提及率不等于购买重要性。', '']
    for item in items:
        with st.expander(item['direction'], expanded=True):
            st.write('依据：' + item['basis'])
            st.write('验证：' + item['validation'])
            st.write('取舍：' + item['tradeoff'])
            evidence = [x for x in coded if x['opinion_id'] in item['evidence_ids']]
            st.dataframe(pd.DataFrame(evidence)[['review_id', 'opinion_id', 'evidence']], use_container_width=True)
        report.extend(['## ' + item['direction'], item['basis'], '验证：' + item['validation'], '取舍：' + item['tradeoff']])
        report.extend(f"> 评论{x['review_id']} / 观点{x['opinion_id']}：{x['evidence']}" for x in evidence)
    st.download_button('下载观点明细 CSV', edited.to_csv(index=False).encode('utf-8-sig'), 'opinions.csv')
    st.download_button('下载统计 CSV', stats.to_csv(index=False).encode('utf-8-sig'), 'statistics.csv')
    st.download_button('下载完整结果 JSON', json.dumps({'extraction': st.session_state['extraction'], 'coded': coded, 'statistics': stats.to_dict('records'), 'opportunities': items, 'model': name, 'base_url': base, 'api_mode': api_mode, 'reasoning_effort': reasoning_effort}, ensure_ascii=False, indent=2), 'analysis.json')
    if items:
        st.download_button('下载开品报告 Markdown', '\n\n'.join(report), 'report.md')
elif 'coded' in st.session_state:
    st.info('当前没有可归并的观点；请先检查模型抽取结果或扩大模型分析范围。')

st.divider()
st.header('业务目标验证工作台')
st.caption('参考 XHS Business Idea Validator 的研究流程，扩展目标相关性、场景、替代方案、分层标签和综合报告。数据来自当前看板筛选后的评价。')
current = st.session_state.get('current_records', [])
workflow_key = json.dumps([st.session_state.get('input_signature', ''), goal_key, base, name, api_mode, reasoning_effort], ensure_ascii=False)
if st.session_state.get('workflow_key') != workflow_key:
    for field in ['signals', 'research_report', 'taxonomy']:
        st.session_state.pop(field, None)
    st.session_state['workflow_key'] = workflow_key
checkpoint_root = Path(__file__).parent / 'outputs' / 'runs'
if st.button('4. 分析目标相关性与研究线索', disabled=not key or not current or not goal_key):
    bar = st.progress(0.0)
    with st.spinner('判断相关性，抽取有原文证据的研究线索…'):
        signal_rows, signal_errors = analyze_signals(build_model(), current, goal_key, Path(__file__).parent / 'outputs' / 'signal_cache', lambda n, total: bar.progress(n / total))
    st.session_state['signals'] = {'rows': signal_rows, 'errors': signal_errors}
    st.session_state.pop('research_report', None)
    folder = save_checkpoint(checkpoint_root, workflow_key, 'signals', st.session_state['signals'])
    st.caption('阶段结果已保存：' + str(folder))

if 'signals' in st.session_state:
    signal_data = st.session_state['signals']
    signal_rows = signal_data['rows']
    if signal_data['errors']:
        st.warning(f"研究线索分析失败 {len(signal_data['errors'])} 条，未纳入综合分析。")
        st.dataframe(signal_data['errors'])
    if signal_rows:
        st.bar_chart(pd.DataFrame(signal_rows)['relevance'].value_counts().rename('文本数'))
        filter_value = st.selectbox('筛选目标相关性', ['全部', '直接相关', '间接相关', '无关', '无法判断'])
        filtered = [x for x in signal_rows if filter_value == '全部' or x['relevance'] == filter_value]
        st.dataframe(pd.DataFrame([{k: v for k, v in x.items() if k != 'signals'} for x in filtered]), use_container_width=True)
        flat = [{**s, 'review_id': r['review_id'], 'relevance': r['relevance']} for r in filtered for s in r['signals']]
        if flat:
            st.dataframe(pd.DataFrame(flat), use_container_width=True)
    st.download_button('下载研究线索 JSON', json.dumps(signal_data, ensure_ascii=False, indent=2), 'research_signals.json')
    st.caption('综合报告仅使用直接或间接相关的证据，最多120条线索和80条相关观点。相关性判断仍需人工抽查。')
    if st.button('5. 生成三级标签与综合验证报告', disabled=not key or not signal_rows):
        try:
            source_coded = st.session_state.get('coded', [])
            related_ids = {x['review_id'] for x in signal_rows if x['relevance'] in ['直接相关', '间接相关']}
            related_coded = [x for x in source_coded if x['review_id'] in related_ids]
            related_denominator = sum(x['kind'] == '消费体验' and x['review_id'] in related_ids for x in st.session_state.get('extraction', {}).get('reviews', []))
            related_stats = statistics(related_coded, related_denominator)
            with st.status('生成分层标签与综合报告…', expanded=True) as status:
                tags = hierarchy(build_model(), related_coded) if related_coded else []
                st.write('三级标签已完成；没有消费观点时仅输出研究线索报告。')
                save_checkpoint(checkpoint_root, workflow_key, 'taxonomy', tags)
                combined = synthesize(build_model(), goal_key, signal_rows, related_coded, related_stats)
                bundle = {'goal': goal_key, 'report': combined, 'tags': tags, 'statistics': related_stats.to_dict('records'), 'model': name, 'base_url': base, 'api_mode': api_mode, 'reasoning_effort': reasoning_effort}
                folder = save_checkpoint(checkpoint_root, workflow_key, 'report', bundle)
                rendered = html_report(goal_key, combined, related_stats, tags)
                (folder / 'report.html').write_text(rendered, encoding='utf-8')
                st.session_state['research_report'] = {'bundle': bundle, 'html': rendered, 'coded_signature': json.dumps(source_coded, ensure_ascii=False, sort_keys=True)}
                status.update(label='报告完成，证据引用校验通过', state='complete')
        except Exception as exc:
            st.error('综合报告生成失败：' + str(exc))

if 'research_report' in st.session_state:
    saved = st.session_state['research_report']
    actual_signature = json.dumps(st.session_state.get('coded', []), ensure_ascii=False, sort_keys=True)
    if saved['coded_signature'] != actual_signature:
        st.warning('需求分类已变化，请重新生成综合报告。下方保留的是上次生成版本。')
    bundle = saved['bundle']
    combined = bundle['report']
    st.subheader('综合验证报告')
    st.write(combined['summary'])
    st.caption('报告证据覆盖：' + json.dumps(combined['coverage'], ensure_ascii=False))
    tabs = st.tabs(['分层标签', '痛点', '场景与人群', '竞品与替代方案', '开品假设', '待补数据'])
    with tabs[0]:
        if bundle['tags']:
            st.dataframe(pd.DataFrame(bundle['tags']), use_container_width=True)
        else:
            st.info('尚无可归并的相关消费观点。可先完成观点抽取与需求归并。')
    for tab, kind in zip(tabs[1:5], ['痛点', '场景与人群', '竞品与替代方案', '开品假设']):
        with tab:
            findings = [x for x in combined['findings'] if x['kind'] == kind]
            if not findings:
                st.info('当前证据不足，未生成该类结论。')
            for finding in findings:
                with st.expander(finding['title'], expanded=True):
                    st.write(finding['interpretation'])
                    st.write('验证：' + finding['next_test'])
                    st.write('取舍：' + finding['tradeoff'])
                    st.dataframe(pd.DataFrame([x for x in combined['evidence'] if x['id'] in finding['evidence_ids']]), use_container_width=True)
    with tabs[5]:
        for gap in combined['gaps']:
            st.write('• ' + gap)
    st.download_button('下载可搜索、可折叠的网页报告', saved['html'], 'research_report.html', mime='text/html')
    st.download_button('下载综合报告 JSON', json.dumps(bundle, ensure_ascii=False, indent=2), 'research_report.json')
