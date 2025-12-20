#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List, Tuple


Pose = Tuple[float, float, float]


def _parse_value(s: str) -> float:
    if not s or s[0] != "s":
        raise ValueError(f"valor sem prefixo 's': {s!r}")
    return float(s[1:])


def _parse_id(s: str) -> Tuple[int, int]:
    if len(s) < 5 or s[3] != "_":
        raise ValueError(f"id invalido: {s!r}")
    n = int(s[:3])
    idx = int(s[4:])
    return n, idx


def load_submission(path: str) -> Dict[int, List[Pose]]:
    by_n: Dict[int, List[Pose]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[:4] != ["id", "x", "y", "deg"]:
            raise ValueError("header invalido (esperado: id,x,y,deg)")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"linha invalida: {row!r}")
            n, idx = _parse_id(row[0])
            x = _parse_value(row[1])
            y = _parse_value(row[2])
            deg = _parse_value(row[3])
            if n not in by_n:
                by_n[n] = [(float("nan"), float("nan"), float("nan"))] * n
            if idx < 0 or idx >= n:
                raise ValueError(f"idx fora de [0,n): {row[0]!r}")
            by_n[n][idx] = (x, y, deg)
    for n, poses in by_n.items():
        if len(poses) != n:
            raise ValueError(f"n={n} inconsistente no input")
        for i, p in enumerate(poses):
            if not all(map(lambda v: v == v, p)):
                raise ValueError(f"faltou id {n:03d}_{i}")
    return by_n


def choose_drop_index(poses: List[Pose], mode: str) -> int:
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if mode == "central":
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        best = None
        for i, (x, y, _) in enumerate(poses):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if best is None or d2 < best[0] or (d2 == best[0] and i < best[1]):
                best = (d2, i)
        return best[1] if best else 0

    # mode == "edge"
    best = None
    for i, (x, y, _) in enumerate(poses):
        margin = min(x - min_x, max_x - x, y - min_y, max_y - y)
        if best is None or margin < best[0] or (margin == best[0] and i < best[1]):
            best = (margin, i)
    return best[1] if best else 0


def fmt_value(v: float, decimals: int) -> str:
    return f"s{v:.{decimals}f}"


def write_submission(path: str,
                     by_n: Dict[int, List[Pose]],
                     n_src: int,
                     poses_n1: List[Pose],
                     decimals: int,
                     n_only: bool) -> None:
    n1 = n_src - 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        if n_only:
            for i, (x, y, deg) in enumerate(poses_n1):
                writer.writerow([f"{n1:03d}_{i}", fmt_value(x, decimals), fmt_value(y, decimals), fmt_value(deg, decimals)])
            return

        for n in sorted(by_n.keys()):
            if n == n1:
                for i, (x, y, deg) in enumerate(poses_n1):
                    writer.writerow([f"{n1:03d}_{i}", fmt_value(x, decimals), fmt_value(y, decimals), fmt_value(deg, decimals)])
                continue
            poses = by_n[n]
            for i, (x, y, deg) in enumerate(poses):
                writer.writerow([f"{n:03d}_{i}", fmt_value(x, decimals), fmt_value(y, decimals), fmt_value(deg, decimals)])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Cria um seed para N-1 removendo uma arvore do layout N. "
            "Opcionalmente roda um comando de refinamento (blend_repair)."
        )
    )
    ap.add_argument("--input", required=True, help="CSV de entrada (submission).")
    ap.add_argument("--output", required=True, help="CSV de saida (seed N-1).")
    ap.add_argument("--n", type=int, required=True, help="N base (ex.: 40).")
    ap.add_argument(
        "--mode",
        choices=["central", "edge"],
        default="central",
        help="Criterio de remocao: central (mais central) ou edge (mais perto da borda).",
    )
    ap.add_argument("--drop-idx", type=int, default=None, help="Idx fixo para remover.")
    ap.add_argument("--decimals", type=int, default=9, help="Casas decimais de saida.")
    ap.add_argument("--n-only", action="store_true", help="Escreve apenas o N-1 derivado.")
    ap.add_argument(
        "--refine-cmd",
        default=None,
        help=(
            "Comando opcional para refinar o seed. Placeholders: {input} {output} {n} {n1}."
        ),
    )
    ap.add_argument(
        "--refine-output",
        default=None,
        help="CSV final do refinamento (default: <output> com sufixo _refined).",
    )
    args = ap.parse_args()

    if args.n <= 1:
        raise SystemExit("--n precisa ser >= 2.")

    by_n = load_submission(args.input)
    if args.n not in by_n:
        raise SystemExit(f"n={args.n} nao encontrado no input.")

    poses = by_n[args.n]
    if args.drop_idx is None:
        drop_idx = choose_drop_index(poses, args.mode)
    else:
        drop_idx = args.drop_idx
    if drop_idx < 0 or drop_idx >= len(poses):
        raise SystemExit(f"drop-idx fora de [0,{len(poses) - 1}].")

    poses_n1 = [p for i, p in enumerate(poses) if i != drop_idx]
    if len(poses_n1) != args.n - 1:
        raise SystemExit("falha ao gerar N-1.")

    write_submission(args.output, by_n, args.n, poses_n1, args.decimals, args.n_only)

    if args.refine_cmd:
        refine_out = args.refine_output
        if refine_out is None:
            root, ext = os.path.splitext(args.output)
            refine_out = f"{root}_refined{ext or '.csv'}"
        cmd = args.refine_cmd.format(
            input=args.output, output=refine_out, n=args.n, n1=args.n - 1
        )
        print(f"Rodando: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"Refinado salvo em: {refine_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
