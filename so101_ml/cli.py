"""python -m so101_ml {train,dagger-finetune,convert,compare-eval,sheet,inspect}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _dataset_dir(name: str) -> Path:
    from so101_ml.paths import project_root

    return project_root() / "data" / "datasets" / name


def cmd_inspect(args: argparse.Namespace) -> int:
    from so101_ml.dataset import SO101Dataset

    root = Path(args.dataset_dir) if args.dataset_dir else _dataset_dir(args.dataset)
    ds = SO101Dataset(root)
    print(json.dumps(ds.meta.to_dict(), indent=2))
    for i in range(ds.meta.episode_count):
        ep = ds.load_episode(i)
        print(
            f"episode {i:06d}  frames={ep.num_frames}  source={ep.source}  "
            f"intervention={ep.intervention}"
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from so101_ml.train import train_from_env

    train_from_env(
        dataset_name=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        seed=args.seed,
        eval_every=args.eval_every,
        backend=args.backend,
        corrections_name=args.corrections,
        init_checkpoint=args.init_checkpoint,
    )
    return 0


def cmd_dagger_finetune(args: argparse.Namespace) -> int:
    from so101_ml.dagger_offline import finetune_act_with_corrections
    from so101_ml.paths import project_root

    root = project_root()
    demos = Path(args.demos) if args.demos else _dataset_dir(args.dataset)
    corrections = (
        Path(args.corrections_dir)
        if args.corrections_dir
        else _dataset_dir(args.corrections or f"{args.dataset}_dagger")
    )
    init = Path(args.init_checkpoint) if args.init_checkpoint else root / "outputs" / "train" / f"act_{args.dataset}"
    out = Path(args.output_dir) if args.output_dir else root / "outputs" / "train" / f"act_{args.dataset}_dagger"
    finetune_act_with_corrections(
        demos=demos,
        corrections=corrections,
        output_dir=out,
        init_checkpoint=init,
        steps=args.steps,
        batch_size=args.batch_size or 2,
        val_frac=args.val_frac,
        seed=args.seed,
        eval_every=args.eval_every,
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from so101_ml.convert_lerobot import convert_so101_to_lerobot
    from so101_ml.paths import project_root

    root = project_root()
    src = Path(args.dataset_dir) if args.dataset_dir else _dataset_dir(args.dataset)
    dest = Path(args.output_dir) if args.output_dir else root / "data" / "lerobot" / args.dataset
    meta = convert_so101_to_lerobot(
        src,
        dest,
        repo_id=args.repo_id or f"local/{args.dataset}",
        max_episodes=args.max_episodes,
    )
    print(json.dumps(meta, indent=2))
    return 0


def cmd_compare_eval(args: argparse.Namespace) -> int:
    from so101_ml.metrics.compare_eval import compare_summaries, format_comparison, load_summary

    report = compare_summaries(load_summary(args.baseline), load_summary(args.candidate))
    print(format_comparison(report), end="")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    from so101_ml.metrics.lab_sheet import export_from_env

    export_from_env(output_dir=args.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="so101_ml",
        description="SO-101 visuomotor ML (offline): ACT, SmolVLA convert, DAgger fine-tune, metrics",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    insp = sub.add_parser("inspect", help="Print dataset meta")
    insp.add_argument("--dataset", default="so101_d405_demo")
    insp.add_argument("--dataset-dir", default="")
    insp.set_defaults(func=cmd_inspect)

    tr = sub.add_parser("train", help="Train or fine-tune ACT")
    tr.add_argument("--dataset", default="so101_d405_demo")
    tr.add_argument("--steps", type=int, default=40_000)
    tr.add_argument("--batch-size", type=int, default=0)
    tr.add_argument("--output-dir", default="")
    tr.add_argument("--val-frac", type=float, default=0.2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--eval-every", type=int, default=1000)
    tr.add_argument("--backend", choices=("torch", "jax"), default="torch")
    tr.add_argument("--corrections", default="", help="DAgger dataset name under data/datasets/")
    tr.add_argument("--init-checkpoint", default="")
    tr.set_defaults(func=cmd_train)

    dg = sub.add_parser("dagger-finetune", help="Combine demos + corrections and fine-tune ACT")
    dg.add_argument("--dataset", default="so101_d405_demo")
    dg.add_argument("--demos", default="", help="Path to demo dataset root")
    dg.add_argument("--corrections", default="", help="Corrections dataset name")
    dg.add_argument("--corrections-dir", default="", help="Path to corrections dataset root")
    dg.add_argument("--init-checkpoint", default="")
    dg.add_argument("--output-dir", default="")
    dg.add_argument("--steps", type=int, default=8_000)
    dg.add_argument("--batch-size", type=int, default=2)
    dg.add_argument("--val-frac", type=float, default=0.2)
    dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--eval-every", type=int, default=500)
    dg.set_defaults(func=cmd_dagger_finetune)

    conv = sub.add_parser("convert", help="Convert local episodes → LeRobot (SmolVLA)")
    conv.add_argument("--dataset", default="so101_d405_demo")
    conv.add_argument("--dataset-dir", default="")
    conv.add_argument("--output-dir", default="")
    conv.add_argument("--repo-id", default="")
    conv.add_argument("--max-episodes", type=int, default=None)
    conv.set_defaults(func=cmd_convert)

    cmp = sub.add_parser("compare-eval", help="Diff two protocol summary.json files")
    cmp.add_argument("--baseline", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--output", default="")
    cmp.set_defaults(func=cmd_compare_eval)

    sheet = sub.add_parser("sheet", help="Export lab workbook Excel + CSV")
    sheet.add_argument("--output-dir", default="")
    sheet.set_defaults(func=cmd_sheet)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
