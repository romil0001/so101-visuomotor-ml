"""Lab workbook export from datasets and eval trial logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from so101_ml.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101_ml.metrics.lab_sheet import collect_all, export_lab_sheet, write_csv_dir


def _demo_dataset(root: Path) -> None:
    ds = SO101Dataset.create(
        root / "data" / "datasets" / "unit_demo",
        DatasetMeta(
            name="unit_demo",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_width=32,
            camera_height=24,
            task="Place keys into metal cup",
            episode_count=0,
        ),
    )
    writer = EpisodeWriter(ds, 0, source="teleop")
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    writer.add_frame(np.zeros(6, np.float32), np.ones(6, np.float32), 0.0, img)
    writer.save()


def _eval_run(root: Path) -> None:
    run = root / "outputs" / "eval" / "act_unit" / "20260101_000000"
    run.mkdir(parents=True)
    (run / "protocol.json").write_text(
        json.dumps(
            {
                "task": "Place keys into metal cup",
                "timeout_s": 60,
                "n_trials_per_condition": 1,
                "conditions": [{"id": "keys=left"}, {"id": "keys=center"}],
            }
        )
    )
    for i, (cid, ok) in enumerate([("keys=left", True), ("keys=center", False)], start=1):
        folder = run / "trials" / f"{i:04d}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(
            json.dumps(
                {
                    "trial_index": i,
                    "condition_id": cid,
                    "success": ok,
                    "steps": 12,
                    "early_stop": False,
                    "timeout_s": 60.0,
                }
            )
        )


def test_collect_and_csv(tmp_path: Path) -> None:
    _demo_dataset(tmp_path)
    _eval_run(tmp_path)
    tables = {t.name: t for t in collect_all(tmp_path)}
    assert tables["Datasets"].rows[0][0] == "unit_demo"
    assert tables["Datasets"].rows[0][1] == 1
    assert tables["Episodes"].rows[0][4] == "teleop"
    assert any(r[0] == "act" for r in tables["Eval_runs"].rows)
    assert len(tables["Eval_trials"].rows) == 2
    paths = write_csv_dir(list(tables.values()), tmp_path / "csv")
    assert (tmp_path / "csv" / "Overview.csv").is_file()
    assert len(paths) >= 5


def test_export_xlsx(tmp_path: Path) -> None:
    import pytest

    openpyxl = pytest.importorskip("openpyxl")
    _demo_dataset(tmp_path)
    _eval_run(tmp_path)
    manifest = export_lab_sheet(root=tmp_path, output_dir=tmp_path / "lab_sheet")
    assert Path(manifest["xlsx"]).is_file()
    wb = openpyxl.load_workbook(manifest["xlsx"])
    assert "Overview" in wb.sheetnames
    assert "Eval_trials" in wb.sheetnames
