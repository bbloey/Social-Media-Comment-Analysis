# 消费者需求研究 Agent

本地 Python + Streamlit 应用，使用可维护的特征知识库和唯一证据引擎，不改动用户源数据。模型仅用于选择性增强；没有 API 时仍可完成规则看板、证据导出和人工复核。

现支持 CSV 及 Excel。默认使用上传文件或演示数据；也可通过 `REVIEW_AGENT_BUSINESS_FILE` 和 `REVIEW_AGENT_TEST_FILE` 预配置本地文件。支持研究目标、相关性筛选、场景/替代方案线索、三级需求标签及可搜索的 HTML 综合报告。详见 [融合说明](INTEGRATION.md)。

页面现在采用“预处理 → 经营看板 → AI需求研究”顺序。预处理支持读取范围、字段映射、空值清理、评论身份去重和保留最新月份；经营看板支持月份、平台、品牌、价格联动筛选，总体/平台词云、品牌结构、价格带、价格分布、评价趋势和明细导出。月份仅作筛选与样本量展示，不自动解释为需求趋势。

经营看板还提供离线多标签特征短语抽取。唯一证据引擎输出原文证据、位置、子句、标准标签、情感、语句类型、置信度、知识库 Hash 与抽取器版本；业务品类直接使用映射字段。页面提供情感分布、待判定池、快速抽查和金标准完整标注。快速抽查不能计算正式召回率；完整标注全部证据后才能输出精确率、召回率和 F1。

## 启动

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

也可在 PowerShell 中运行 `./start.ps1`。打开终端显示的本地地址。

## 在线示例部署

仓库包含可直接用于 Streamlit Community Cloud 的英文示例目录：

```text
examples/streamlit-dashboard/
```

部署时选择：

- Repository：`bbloey/Social-Media-Comment-Analysis`
- Branch：`main`
- Main file path：`examples/streamlit-dashboard/app.py`

示例目录提供合成评论数据，不包含真实消费者或业务数据。详细步骤见 [Streamlit Dashboard Demo](examples/streamlit-dashboard/README.md)。离线规则看板无需 API Key；模型能力为可选配置，真实密钥不得提交到 GitHub。

Python 3.12 需要复现本次测试版本时，可安装 `requirements-lock-py312.txt`。

侧栏默认使用 GPT-5.6 Sol 和 OpenAI Responses API，也保留兼容 Chat Completions 模式。可通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 等环境变量设置；密钥不保存到项目文件。

上传 CSV（UTF-8 或 GB18030），选择评论列，先处理30条。现有 cleaned_data_merged.csv 可选 post_content 列。也可勾选示例评论试跑。

依次点击抽取、归并、生成开品假设。归并后可编辑 category 列合并同义类别；修改后需重新生成报告。结果支持 CSV、JSON 和 Markdown 下载。

## 分析逻辑与边界

- 开放抽取具体属性、观点、情绪、原文证据、明确诉求及推断需求；一条评论允许多个观点。
- 广告、无关及信息不足内容不进入体验统计。该判断需人工抽查，短文本可能误判。
- 核验原文证据确实存在；不保证语义抽取绝对正确。建议人工审阅50—100条，并检查漏抽观点、错判极性和无依据推断。
- 归并每批40条，复用已发现分类并允许新增；语义结构校验失败时自动修复一次，仍失败则交还用户复核。批次顺序可能影响分类，需校订。
- 提及率分母是成功识别为消费体验的去重评论数；含负面率分母是该类提及评论数，混合评价计入含负面。去重会改变原始评论分布。
- 机会生成使用提及数最高的20类、每类最多8条证据，属于探索性研究，未覆盖全部长尾。方向是待验证假设，不能据此直接推断购买重要性或市场规模。
- 当前读取CSV前10000行，固定种子随机抽样，单次最多2000条；不等于对完整大文件随机抽样。没有星级/品牌/时间分层抽样，不能做因果分析。
- 所选文本会发送到配置的模型服务，接口调用产生费用；先用小样本验证。不得将HTTP服务暴露到公网。
- outputs/cache 保存按文本、接口、模型和提示词哈希索引的成功抽取，可断点复用；其中包含原文证据。删除此目录可清除缓存。失败条目单独展示，不默默纳入分母。
- 开品报告只允许引用实际观点ID；原文属于不可信数据，提示词要求模型不执行评论中的指令。

## 验证

```powershell
python -m unittest discover -s tests -v
```

离线测试覆盖上下文极性、否定、诉求、转折、多标签去重、知识库版本、金标准指标、归并修复和报告证据引用。真实模型调用需要有效接口与密钥后试跑。
