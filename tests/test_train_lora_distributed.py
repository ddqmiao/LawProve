from scripts import train_lora


def test_read_jsonl_honors_limit_without_reading_remaining_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id":1}\n{"id":2}\nnot-json\n', encoding="utf-8")

    rows = train_lora.read_jsonl(path, limit=2)

    assert rows == [{"id": 1}, {"id": 2}]


def test_model_device_map_uses_auto_for_single_process():
    kwargs = train_lora.model_load_kwargs(
        torch_dtype="bf16",
        device_map="auto",
        env={},
    )

    assert kwargs["device_map"] == "auto"


def test_model_device_map_is_omitted_for_torchrun_distributed():
    kwargs = train_lora.model_load_kwargs(
        torch_dtype="bf16",
        device_map="auto",
        env={"WORLD_SIZE": "2", "LOCAL_RANK": "1"},
    )

    assert "device_map" not in kwargs


def test_only_global_rank_zero_writes_monitor_log():
    assert train_lora.should_write_monitor(env={})
    assert train_lora.should_write_monitor(env={"RANK": "0", "WORLD_SIZE": "2"})
    assert not train_lora.should_write_monitor(env={"RANK": "1", "WORLD_SIZE": "2"})
