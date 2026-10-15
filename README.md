# LawProve

Fine-grained grounding of facts and legal provisions in Chinese criminal judgments involving multiple defendants. Given a target defendant, a court-view unit, and candidate facts and legal provisions, LawProve identifies the fact sentences and exact provisions that directly support the court's finding.

```json
{"support_facts": ["F1", "F2"], "support_laws": ["67-3"]}
```

Pipeline: case-level out-of-fold (OOF) hybrid retrieval → listwise candidate selection → Qwen3 LoRA fine-tuning with joint generation and provision ranking → structured prediction and evaluation.

| Directory | Contents |
|---|---|
| `law_grounding/` | Legal provision normalization, retrieval, candidate construction, joint ranking, and evaluation |
| `scripts/` | Entry points for data preparation, training, inference, and evaluation |
| `data_pipeline/` | Case cleaning, sentence segmentation, legal text parsing, and model-assisted annotation |
| `tests/` | Tests for data processing, retrieval, training interfaces, and evaluation metrics |

Requires Python 3.10 or later. Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Specify model and data paths through command-line arguments. Use `--help` to view the options for each entry point:

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

For model-assisted annotation, copy `data_pipeline/.env.example` to `data_pipeline/.env.local` and fill in the API configuration.

Run the tests:

```bash
python -m pytest -q
```
