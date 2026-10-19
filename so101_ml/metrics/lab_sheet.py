"""Export lab readings and performance into Excel + CSV for Google Sheets.

Scans data/datasets/, outputs/train/, and outputs/eval/. Train loss is kept as
a diagnostic column; protocol pc_success is the reported manipulation score.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from so101_ml.metrics.publish_baseline import load_trial_results, policy_family, summary_from_trials

SHEET_ORDER = (
    "Overview",
    "Datasets",
    "Episodes",
    "Training",
    "Eval_runs",
    "Eval_by_condition",
    "Eval_trials",
    "Hardware_log",
    "DAgger_notes",
)


@dataclass(frozen=True)
class SheetTable:
    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _last_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        return {}
    last: dict[str, Any] = {}
    best_val: float | None = None
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            last = row
            if "val_l1" in row:
                v = float(row["val_l1"])
                if best_val is None or v < best_val:
                    best_val = v
    out = dict(last)
    if best_val is not None:
        out["best_val_l1"] = best_val
    return out


def collect_datasets(root: Path) -> SheetTable:
    headers = (
        "dataset",
        "episode_count",
        "fps",
        "task",
        "cameras",
        "use_depth",
        "depth_camera",
        "image_w",
        "image_h",
        "path",
        "notes",
    )
    rows: list[tuple[Any, ...]] = []
    ds_root = root / "data" / "datasets"
    if ds_root.is_dir():
        for meta_path in sorted(ds_root.glob("*/meta.json")):
            meta = _read_json(meta_path) or {}
            name = str(meta.get("name") or meta_path.parent.name)
            cameras = meta.get("camera_names") or [meta.get("camera_name", "")]
            note = ""
            if name.endswith("_dagger") or "dagger" in name:
                note = "DAgger corrections"
            elif "__with_dagger" in name:
                note = "Merged demos + corrections"
            rows.append(
                (
                    name,
                    int(meta.get("episode_count", 0) or 0),
                    int(meta.get("fps", 0) or 0),
                    str(meta.get("task", "")),
                    ",".join(str(c) for c in cameras),
                    bool(meta.get("use_depth", False)),
                    str(meta.get("depth_camera_name", "")),
                    int(meta.get("camera_width", 0) or 0),
                    int(meta.get("camera_height", 0) or 0),
                    _rel(meta_path.parent, root),
                    note,
                )
            )
    return SheetTable("Datasets", headers, tuple(rows))


def collect_episodes(root: Path) -> SheetTable:
    headers = (
        "dataset",
        "episode",
        "num_frames",
        "duration_s",
        "source",
        "intervention",
        "task",
        "path",
    )
    rows: list[tuple[Any, ...]] = []
    ds_root = root / "data" / "datasets"
    if not ds_root.is_dir():
        return SheetTable("Episodes", headers, ())
    for meta_path in sorted(ds_root.glob("*/meta.json")):
        meta = _read_json(meta_path) or {}
        fps = max(int(meta.get("fps", 0) or 0), 1)
        dataset = str(meta.get("name") or meta_path.parent.name)
        episodes_dir = meta_path.parent / "episodes"
        if not episodes_dir.is_dir():
            continue
        for ep_json in sorted(episodes_dir.glob("*/episode.json")):
            emeta = _read_json(ep_json) or {}
            n = int(emeta.get("num_frames", 0) or 0)
            rows.append(
                (
                    dataset,
                    int(emeta.get("index", -1)),
                    n,
                    round(n / fps, 3),
                    str(emeta.get("source", "teleop")),
                    bool(emeta.get("intervention", False)),
                    str(emeta.get("task", meta.get("task", ""))),
                    _rel(ep_json.parent, root),
                )
            )
    return SheetTable("Episodes", headers, tuple(rows))


def collect_training(root: Path) -> SheetTable:
    headers = (
        "job",
        "backend",
        "seed",
        "steps_configured",
        "last_step",
        "last_loss",
        "last_l1",
        "last_kl",
        "best_val_l1",
        "val_frac",
        "use_depth",
        "cameras",
        "task",
        "n_train_episodes",
        "n_val_episodes",
        "force_train",
        "init_checkpoint",
        "has_best_val",
        "has_last",
        "path",
    )
    rows: list[tuple[Any, ...]] = []
    train_root = root / "outputs" / "train"
    if not train_root.is_dir():
        return SheetTable("Training", headers, ())
    for job_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        cfg = _read_json(job_dir / "config.json") or {}
        split = _read_json(job_dir / "split.json") or {}
        metrics = _last_metrics(job_dir / "metrics.jsonl")
        cameras = cfg.get("camera_names") or []
        if isinstance(cameras, str):
            cameras = [cameras]
        rows.append(
            (
                job_dir.name,
                str(cfg.get("backend", "")),
                cfg.get("seed", ""),
                cfg.get("steps", ""),
                metrics.get("step", ""),
                metrics.get("loss", ""),
                metrics.get("l1", ""),
                metrics.get("kl", ""),
                metrics.get("best_val_l1", metrics.get("val_l1", "")),
                cfg.get("val_frac", ""),
                bool(cfg.get("use_depth", False)),
                ",".join(str(c) for c in cameras),
                str(cfg.get("task", "")),
                len(split.get("train_episodes") or []),
                len(split.get("val_episodes") or []),
                ",".join(str(i) for i in (split.get("force_train") or cfg.get("force_train") or [])),
                str(cfg.get("init_checkpoint", "")),
                (job_dir / "best_val" / "model.pt").is_file(),
                (job_dir / "last" / "model.pt").is_file() or (job_dir / "model.pt").is_file(),
                _rel(job_dir, root),
            )
        )
    return SheetTable("Training", headers, tuple(rows))


def _run_id_from_path(run_dir: Path) -> str:
    return run_dir.name


def collect_eval(root: Path) -> tuple[SheetTable, SheetTable, SheetTable]:
    run_headers = (
        "policy_family",
        "run_id",
        "pc_success",
        "n_success",
        "n_trials",
        "complete_protocol",
        "task",
        "timeout_s",
        "n_trials_per_condition",
        "checkpoint_hint",
        "path",
    )
    cond_headers = (
        "policy_family",
        "run_id",
        "condition_id",
        "n",
        "n_success",
        "pc_success",
        "path",
    )
    trial_headers = (
        "policy_family",
        "run_id",
        "trial_index",
        "condition_id",
        "success",
        "steps",
        "early_stop",
        "timeout_s",
        "path",
    )
    run_rows: list[tuple[Any, ...]] = []
    cond_rows: list[tuple[Any, ...]] = []
    trial_rows: list[tuple[Any, ...]] = []
    eval_root = root / "outputs" / "eval"
    if not eval_root.is_dir():
        return (
            SheetTable("Eval_runs", run_headers, ()),
            SheetTable("Eval_by_condition", cond_headers, ()),
            SheetTable("Eval_trials", trial_headers, ()),
        )

    run_dirs: set[Path] = set()
    for trials in eval_root.rglob("trials"):
        if trials.is_dir():
            run_dirs.add(trials.parent)
    for summary in eval_root.rglob("summary.json"):
        run_dirs.add(summary.parent)

    for run_dir in sorted(run_dirs, key=lambda p: str(p)):
        family = policy_family(run_dir)
        run_id = _run_id_from_path(run_dir)
        protocol = _read_json(run_dir / "protocol.json") or {}
        results = load_trial_results(run_dir)
        summary = _read_json(run_dir / "summary.json")
        if summary is None and results:
            summary = summary_from_trials(results)
        if summary is None:
            continue
        n_trials = int(summary.get("n_trials", len(results)) or 0)
        n_success = int(summary.get("n_success", 0) or 0)
        pc = float(summary.get("pc_success", 0.0) or 0.0)
        expected = None
        if protocol.get("n_trials_per_condition") and protocol.get("conditions"):
            expected = int(protocol["n_trials_per_condition"]) * len(protocol["conditions"])
        complete = bool(expected is not None and n_trials >= expected)
        parent_hint = run_dir.parent.name
        run_rows.append(
            (
                family,
                run_id,
                pc,
                n_success,
                n_trials,
                complete,
                str(protocol.get("task", "")),
                protocol.get("timeout_s", ""),
                protocol.get("n_trials_per_condition", ""),
                parent_hint,
                _rel(run_dir, root),
            )
        )
        per = summary.get("per_condition") or summary.get("by_condition") or {}
        for cid, row in per.items():
            if not isinstance(row, dict):
                continue
            n = int(row.get("n", row.get("trials", 0)) or 0)
            if n <= 0:
                continue
            cond_rows.append(
                (
                    family,
                    run_id,
                    str(cid),
                    n,
                    int(row.get("n_success", row.get("successes", 0)) or 0),
                    float(row.get("pc_success", 0.0) or 0.0),
                    _rel(run_dir, root),
                )
            )
        for result in results:
            trial_rows.append(
                (
                    family,
                    run_id,
                    result.trial_index,
                    result.condition_id,
                    bool(result.success),
                    result.steps,
                    bool(result.early_stop),
                    result.timeout_s,
                    _rel(run_dir / "trials" / f"{result.trial_index:04d}", root),
                )
            )

    return (
        SheetTable("Eval_runs", run_headers, tuple(run_rows)),
        SheetTable("Eval_by_condition", cond_headers, tuple(cond_rows)),
        SheetTable("Eval_trials", trial_headers, tuple(trial_rows)),
    )


def hardware_log_template() -> SheetTable:
    headers = (
        "date_utc",
        "session",
        "follower_port",
        "leader_port",
        "follower_p",
        "follower_i",
        "follower_d",
        "supply_v",
        "zmq_host",
        "cameras",
        "depth_ok",
        "omen_ok",
        "notes",
    )
    # One blank starter row the operator fills by hand in Sheets/Excel.
    blank = tuple("" for _ in headers)
    return SheetTable("Hardware_log", headers, (blank,))


def dagger_notes_template(root: Path) -> SheetTable:
    headers = (
        "round",
        "date_utc",
        "base_checkpoint",
        "corrections_dataset",
        "n_corrections",
        "finetune_job",
        "baseline_pc_success",
        "candidate_pc_success",
        "delta_pc_success",
        "notes",
    )
    rows: list[tuple[Any, ...]] = []
    ds_root = root / "data" / "datasets"
    if ds_root.is_dir():
        for meta_path in sorted(ds_root.glob("*dagger*/meta.json")):
            name = meta_path.parent.name
            if "__with_dagger" in name:
                continue
            meta = _read_json(meta_path) or {}
            rows.append(
                (
                    "",
                    "",
                    "",
                    name,
                    int(meta.get("episode_count", 0) or 0),
                    "",
                    "",
                    "",
                    "",
                    "Fill delta after compare-eval",
                )
            )
    if not rows:
        rows.append(tuple("" for _ in headers))
    return SheetTable("DAgger_notes", headers, tuple(rows))


def build_overview(
    root: Path,
    datasets: SheetTable,
    training: SheetTable,
    eval_runs: SheetTable,
    episodes: SheetTable,
) -> SheetTable:
    headers = ("field", "value")
    n_ds = len(datasets.rows)
    n_ep = len(episodes.rows)
    n_train = len(training.rows)
    n_eval = len(eval_runs.rows)
    # Prefer a complete grid, then more trials, then higher pc_success.
    act_best: tuple[Any, ...] | None = None
    smol_best: tuple[Any, ...] | None = None
    for row in eval_runs.rows:
        family = row[0]
        key = (1 if row[5] else 0, int(row[4] or 0), float(row[2] or 0))
        if family == "act":
            if act_best is None or key > (1 if act_best[5] else 0, int(act_best[4] or 0), float(act_best[2] or 0)):
                act_best = row
        elif family == "smolvla":
            if smol_best is None or key > (
                1 if smol_best[5] else 0,
                int(smol_best[4] or 0),
                float(smol_best[2] or 0),
            ):
                smol_best = row
    best_act = (
        f"{act_best[2]}% ({act_best[3]}/{act_best[4]}) · {act_best[1]}" if act_best else ""
    )
    best_smolvla = (
        f"{smol_best[2]}% ({smol_best[3]}/{smol_best[4]}) · {smol_best[1]}" if smol_best else ""
    )

    baseline = _read_json(root / "assets" / "eval" / "baseline.json") or {}
    rows = (
        ("generated_at", _utc_now()),
        ("project", "SO-101 Visuomotor Manipulation"),
        ("root", str(root.resolve())),
        ("n_datasets", n_ds),
        ("n_episodes_indexed", n_ep),
        ("n_training_jobs", n_train),
        ("n_eval_runs", n_eval),
        ("published_baseline_fixture", baseline.get("fixture", "")),
        ("act_baseline_pc_success", best_act),
        ("smolvla_baseline_pc_success", best_smolvla),
        ("primary_metric", "protocol pc_success (not train loss)"),
        ("next_step", "DAgger corrections → fine-tune → compare-eval"),
        ("google_sheets", "File → Import → Upload the .xlsx, or import each CSV as a sheet"),
    )
    return SheetTable("Overview", headers, rows)


def collect_all(root: Path) -> list[SheetTable]:
    root = Path(root)
    datasets = collect_datasets(root)
    episodes = collect_episodes(root)
    training = collect_training(root)
    eval_runs, eval_cond, eval_trials = collect_eval(root)
    overview = build_overview(root, datasets, training, eval_runs, episodes)
    return [
        overview,
        datasets,
        episodes,
        training,
        eval_runs,
        eval_cond,
        eval_trials,
        hardware_log_template(),
        dagger_notes_template(root),
    ]


def write_csv_dir(tables: Sequence[SheetTable], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in tables:
        path = out_dir / f"{table.name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(table.headers)
            writer.writerows(table.rows)
        written.append(path)
    return written


def write_xlsx(tables: Sequence[SheetTable], path: Path) -> Path:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required for .xlsx export. Install with: pip install openpyxl"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    # Remove the default sheet after we create the first named one.
    default = wb.active
    first = True
    for table in tables:
        if first:
            ws = default
            ws.title = table.name[:31]
            first = False
        else:
            ws = wb.create_sheet(title=table.name[:31])
        ws.append(list(table.headers))
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in table.rows:
            ws.append(list(row))
        for i, header in enumerate(table.headers, start=1):
            width = min(48, max(10, len(str(header)) + 2))
            if table.rows:
                sample = max(len(str(r[i - 1])) for r in table.rows[:50])
                width = min(48, max(width, sample + 1))
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"
    wb.save(path)
    return path


def export_lab_sheet(
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    from so101_ml.paths import project_root

    root = Path(root) if root else project_root()
    out = Path(output_dir) if output_dir else root / "outputs" / "lab_sheet"
    tables = collect_all(root)
    csv_dir = out / "csv"
    csv_paths = write_csv_dir(tables, csv_dir)
    xlsx_path = out / "SO101_lab_workbook.xlsx"
    write_xlsx(tables, xlsx_path)
    manifest = {
        "generated_at": _utc_now(),
        "root": str(root.resolve()),
        "xlsx": str(xlsx_path.resolve()),
        "csv_dir": str(csv_dir.resolve()),
        "sheets": [
            {"name": t.name, "n_rows": len(t.rows), "n_cols": len(t.headers)} for t in tables
        ],
        "google_sheets": (
            "Upload SO101_lab_workbook.xlsx to Google Drive → Open with Google Sheets, "
            "or create a blank Sheet and File → Import each CSV under csv/."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_from_env(*, output_dir: str = "") -> dict[str, Any]:
    from so101_ml.paths import project_root

    root = project_root()
    out = Path(output_dir) if output_dir else root / "outputs" / "lab_sheet"
    manifest = export_lab_sheet(root=root, output_dir=out)
    print(f"Wrote {manifest['xlsx']}")
    print(f"CSV sheets: {manifest['csv_dir']}")
    print(manifest["google_sheets"])
    return manifest
