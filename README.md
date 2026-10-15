# LawProve

面向多被告裁判文书的细粒度事实与法律依据溯源。输入目标被告、裁判观点及事实与法条候选，输出支撑该观点的事实句 ID 和精确法条 ID。

```json
{"support_facts": ["F1", "F2"], "support_laws": ["67-3"]}
```

主要流程：案件级 OOF 混合检索 → Listwise 候选筛选 → Qwen3 LoRA → 联合生成与法条排序 → 结构化输出与评测。

| 目录 | 内容 |
|---|---|
| `law_grounding/` | 法条规范化、检索、候选构造、联合排序与评测 |
| `scripts/` | 数据准备、训练、推理与评测入口 |
| `data_pipeline/` | 案件清洗、句子切分、法条解析与模型辅助标注 |
| `tests/` | 数据处理、检索、训练接口与指标测试 |

Python 3.10+。在仓库根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

模型和数据路径通过命令行参数指定。主要入口：

```bash
python scripts/prepare_data.py --help
python scripts/evaluate_supervised_oof_retrieval.py --help
python scripts/evaluate_embedding_retrieval.py --help
python scripts/evaluate_hybrid_retrieval.py --help
python scripts/prepare_law_listwise_sft.py --help
python scripts/train_lora.py --help
python scripts/prepare_joint_rank_grounding_data.py --help
python scripts/train_joint_rank_grounding.py --help
python scripts/infer_joint_rank_grounding.py --help
python scripts/infer_qwen.py --help
python scripts/evaluate.py --help
```

模型辅助标注使用 `data_pipeline/.env.example` 中的配置项，可复制为 `data_pipeline/.env.local` 后填写。

```bash
python -m pytest -q
```
