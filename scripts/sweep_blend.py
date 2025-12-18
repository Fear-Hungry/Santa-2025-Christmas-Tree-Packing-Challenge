#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys


_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_KEYS = {"seed", "shift_a", "shift_b", "out", "run"}


def _parse_name_value(spec: str, *, flag: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"{flag}: esperado NAME=VALUE; recebido: {spec!r}")
    name, value = spec.split("=", 1)
    if not name or not _VALID_NAME_RE.match(name):
        raise SystemExit(f"{flag}: nome inválido: {name!r}")
    if name in _RESERVED_KEYS:
        raise SystemExit(f"{flag}: {name!r} é reservado (use outro nome).")
    if value == "":
        raise SystemExit(f"{flag}: valor vazio para {name!r}.")
    return name, value


def _parse_choice(spec: str) -> tuple[str, list[str]]:
    name, value = _parse_name_value(spec, flag="--choice")
    items = [x for x in value.split("|") if x != ""]
    if not items:
        raise SystemExit(f"--choice: lista vazia para {name!r}.")
    return name, items


def _parse_uniform(spec: str) -> tuple[str, float, float]:
    name, value = _parse_name_value(spec, flag="--uniform")
    if "," not in value:
        raise SystemExit(f"--uniform: esperado NAME=LO,HI; recebido: {spec!r}")
    lo_s, hi_s = value.split(",", 1)
    try:
        lo = float(lo_s)
        hi = float(hi_s)
    except ValueError:
        raise SystemExit(f"--uniform: LO/HI inválidos: {spec!r}")
    if not (lo <= hi):
        raise SystemExit(f"--uniform: precisa ter LO <= HI; recebido: {spec!r}")
    return name, lo, hi


def _parse_randint(spec: str) -> tuple[str, int, int]:
    name, value = _parse_name_value(spec, flag="--randint")
    if "," not in value:
        raise SystemExit(f"--randint: esperado NAME=LO,HI; recebido: {spec!r}")
    lo_s, hi_s = value.split(",", 1)
    try:
        lo = int(lo_s)
        hi = int(hi_s)
    except ValueError:
        raise SystemExit(f"--randint: LO/HI inválidos: {spec!r}")
    if not (lo <= hi):
        raise SystemExit(f"--randint: precisa ter LO <= HI; recebido: {spec!r}")
    return name, lo, hi


def _resolve_exe(candidates: list[str]) -> str:
    for c in candidates:
        if ("/" in c or c.startswith(".")) and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        found = shutil.which(c)
        if found:
            return found
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Roda M variantes do solver e faz blend por n (pega o melhor s_n de cada input)."
        )
    )
    ap.add_argument(
        "--cmd",
        action="append",
        required=True,
        help=(
            "Template de comando do solver. Placeholders disponíveis: "
            "{seed} {shift_a} {shift_b} {out} {run} (e quaisquer variáveis definidas por --set/--choice/--uniform/--randint). "
            "Ex.: \"./bin/solver_tile --k 4 --seed {seed} --shift {shift_a},{shift_b} --output {out}\""
        ),
    )
    ap.add_argument(
        "--cmd-mode",
        choices=["cycle", "random"],
        default="cycle",
        help="Se passar múltiplos --cmd, escolhe 'cycle' (round-robin) ou 'random'.",
    )
    ap.add_argument("--runs", type=int, default=10, help="Número de runs (M).")
    ap.add_argument("--seed0", type=int, default=1, help="Seed base (run i usa seed0+i).")
    ap.add_argument(
        "--runs-dir",
        default="runs",
        help="Diretório para salvar os CSVs individuais.",
    )
    ap.add_argument(
        "--out",
        default="submission_sweep_ensemble.csv",
        help="CSV final (ensemble por n).",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Define variável fixa para o template (pode repetir).",
    )
    ap.add_argument(
        "--choice",
        action="append",
        default=[],
        metavar="NAME=V1|V2|V3",
        help="Escolhe aleatoriamente um valor por run (pode repetir; use aspas por causa do '|').",
    )
    ap.add_argument(
        "--uniform",
        action="append",
        default=[],
        metavar="NAME=LO,HI",
        help="Amostra float U[LO,HI] por run (pode repetir).",
    )
    ap.add_argument(
        "--randint",
        action="append",
        default=[],
        metavar="NAME=LO,HI",
        help="Amostra int em [LO,HI] por run (pode repetir).",
    )
    ap.add_argument(
        "--float-digits",
        type=int,
        default=6,
        help="Casas decimais para variáveis float geradas (shift_a/shift_b e --uniform).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Se o CSV do run já existir, reaproveita e não reroda o solver.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Escreve manifest JSONL (default: runs_dir/manifest.jsonl). Use '-' para desabilitar.",
    )
    ap.add_argument(
        "--no-final-rigid",
        action="store_true",
        help="Desliga o pós-processamento final rigid no ensemble.",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="Continua mesmo se algum run falhar.",
    )
    args = ap.parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs precisa ser > 0.")
    if args.float_digits < 0 or args.float_digits > 18:
        raise SystemExit("--float-digits precisa estar em [0, 18].")

    os.makedirs(args.runs_dir, exist_ok=True)

    rng = random.Random(args.seed0 ^ 0x9E3779B97F4A7C15)
    run_paths: list[str] = []

    fixed_vars: dict[str, str] = {}
    choice_vars: dict[str, list[str]] = {}
    uniform_vars: dict[str, tuple[float, float]] = {}
    randint_vars: dict[str, tuple[int, int]] = {}

    def _reserve(name: str) -> None:
        if name in fixed_vars or name in choice_vars or name in uniform_vars or name in randint_vars:
            raise SystemExit(f"Variável duplicada: {name!r}")

    for spec in args.set:
        name, value = _parse_name_value(spec, flag="--set")
        _reserve(name)
        fixed_vars[name] = value
    for spec in args.choice:
        name, items = _parse_choice(spec)
        _reserve(name)
        choice_vars[name] = items
    for spec in args.uniform:
        name, lo, hi = _parse_uniform(spec)
        _reserve(name)
        uniform_vars[name] = (lo, hi)
    for spec in args.randint:
        name, lo, hi = _parse_randint(spec)
        _reserve(name)
        randint_vars[name] = (lo, hi)

    manifest_path = None
    if args.manifest == "-":
        manifest_path = None
    elif args.manifest:
        manifest_path = args.manifest
    else:
        manifest_path = os.path.join(args.runs_dir, "manifest.jsonl")

    for run in range(args.runs):
        seed = args.seed0 + run
        shift_a = rng.random()
        shift_b = rng.random()
        out_path = os.path.join(args.runs_dir, f"run_{run:03d}.csv")

        extra_vars: dict[str, str] = dict(fixed_vars)
        for name, items in choice_vars.items():
            extra_vars[name] = rng.choice(items)
        for name, (lo, hi) in uniform_vars.items():
            extra_vars[name] = f"{rng.uniform(lo, hi):.{args.float_digits}f}"
        for name, (lo, hi) in randint_vars.items():
            extra_vars[name] = str(rng.randint(lo, hi))

        cmd_templates: list[str] = args.cmd
        if args.cmd_mode == "random" and len(cmd_templates) > 1:
            cmd_template = rng.choice(cmd_templates)
        else:
            cmd_template = cmd_templates[run % len(cmd_templates)]

        fmt_vars = {
            "seed": seed,
            "shift_a": f"{shift_a:.{args.float_digits}f}",
            "shift_b": f"{shift_b:.{args.float_digits}f}",
            "out": out_path,
            "run": run,
            **extra_vars,
        }

        try:
            cmd = cmd_template.format(**fmt_vars)
        except KeyError as exc:
            missing = exc.args[0]
            raise SystemExit(f"Placeholder não definido no template: {missing!r}")

        print(f"[{run + 1:03d}/{args.runs:03d}] {cmd}", flush=True)

        resumed = False
        if args.resume and os.path.exists(out_path):
            resumed = True
        else:
            try:
                subprocess.run(["bash", "-lc", cmd], check=True)
            except subprocess.CalledProcessError as exc:
                print(f"Run {run} falhou (exit={exc.returncode}).", file=sys.stderr)
                if not args.keep_going:
                    return exc.returncode
                continue

        if manifest_path is not None:
            rec = {
                "run": run,
                "seed": seed,
                "shift_a": float(fmt_vars["shift_a"]),
                "shift_b": float(fmt_vars["shift_b"]),
                "out": out_path,
                "cmd": cmd,
                "resumed": resumed,
                "vars": extra_vars,
            }
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not os.path.exists(out_path):
            print(f"Run {run} não gerou arquivo: {out_path}", file=sys.stderr)
            if not args.keep_going:
                return 1
            continue

        run_paths.append(out_path)

    if not run_paths:
        print("Nenhum CSV gerado; nada para ensemblar.", file=sys.stderr)
        return 1

    ensemble_bin = _resolve_exe(["./bin/ensemble_submissions", "./ensemble_submissions", "ensemble_submissions"])
    score_bin = _resolve_exe(["./bin/score_submission", "./score_submission", "score_submission"])

    ensemble_cmd = [ensemble_bin, args.out, *run_paths]
    if args.no_final_rigid:
        ensemble_cmd.append("--no-final-rigid")

    subprocess.run(ensemble_cmd, check=True)
    subprocess.run([score_bin, args.out], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
