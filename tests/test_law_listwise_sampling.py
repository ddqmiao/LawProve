import json

from law_grounding.law_listwise_sampling import rebalance_listwise_records


def record(unit: str, group_index: int, *, gold_laws=None, keep=None, maybe=None):
    gold_laws = ["67-3"] if gold_laws is None else gold_laws
    keep = [] if keep is None else keep
    maybe = [] if maybe is None else maybe
    candidates = ["67-3", "25-1"]
    reject = [item for item in candidates if item not in keep and item not in maybe]
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"keep": keep, "maybe": maybe, "reject": reject},
                    ensure_ascii=False,
                ),
            },
        ],
        "meta": {
            "case_id": unit,
            "defendant_index": 0,
            "defendant": "张某",
            "court_unit_id": "CV0",
            "gold_laws": gold_laws,
            "group_index": group_index,
            "candidate_law_ids": candidates,
        },
    }


def test_rebalance_keeps_positives_limits_negatives_and_repeats_late_positives():
    records = [
        record("positive", 0, keep=["67-3"]),
        record("positive", 1),
        record("positive", 2, maybe=["67-3"]),
        record("positive", 3),
    ]

    output, stats = rebalance_listwise_records(
        records,
        max_negative_groups_per_unit=1,
        empty_gold_unit_fraction=0.0,
        late_positive_copies=2,
        seed=7,
    )

    groups = [row["meta"]["group_index"] for row in output]
    assert sorted(groups) == [0, 1, 2, 2]
    assert stats["positive_records"] == 2
    assert stats["late_positive_extra_copies"] == 1
    assert stats["negative_records"] == 1


def test_rebalance_samples_empty_gold_units_deterministically():
    records = [
        record("empty-a", 0, gold_laws=[]),
        record("empty-a", 1, gold_laws=[]),
        record("empty-b", 0, gold_laws=[]),
        record("empty-b", 1, gold_laws=[]),
    ]

    first, stats = rebalance_listwise_records(
        records,
        max_negative_groups_per_unit=1,
        empty_gold_unit_fraction=0.5,
        late_positive_copies=1,
        seed=11,
    )
    second, _ = rebalance_listwise_records(
        records,
        max_negative_groups_per_unit=1,
        empty_gold_unit_fraction=0.5,
        late_positive_copies=1,
        seed=11,
    )

    assert first == second
    assert len(first) == 1
    assert stats["selected_empty_gold_units"] == 1
