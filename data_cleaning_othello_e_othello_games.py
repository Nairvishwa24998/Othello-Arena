#!/usr/bin/env python3

from __future__ import annotations
import argparse, csv, string, sys
from pathlib import Path
import numpy as np

# ── Othello helpers ─────────────────────────────────────────────────────────
DIRS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
FILE2COL = {c: i for i, c in enumerate(string.ascii_lowercase[:8])}


def start_board() -> np.ndarray:
    b = np.full((8, 8), ".", dtype="<U1")
    b[3, 3] = b[4, 4] = "W"
    b[3, 4] = b[4, 3] = "B"
    return b


def flips(board: np.ndarray, r: int, c: int, colour: str) -> list[tuple[int, int]]:
    if board[r, c] != ".":  # occupied
        return []
    opp = "W" if colour == "B" else "B"
    acc: list[tuple[int, int]] = []
    for dx, dy in DIRS:
        path = []
        i, j = r + dx, c + dy
        while 0 <= i < 8 and 0 <= j < 8 and board[i, j] == opp:
            path.append((i, j))
            i += dx
            j += dy
        if path and 0 <= i < 8 and 0 <= j < 8 and board[i, j] == colour:
            acc.extend(path)
    return acc


def token_rc(tok: str) -> tuple[int, int]:
    if len(tok) != 2:
        raise ValueError(f"Bad move token {tok!r}")
    col = FILE2COL.get(tok[0].lower())
    row = int(tok[1]) - 1
    if col is None or not 0 <= row < 8:
        raise ValueError(f"Bad move token {tok!r}")
    return row, col

# method to convert the csv to npz format. The e othello datbase was in csv format only
def csv_to_npz(csv_file: Path, winner_col: str, moves_col: str, dialect: csv.Dialect) -> None:
    states, policies, values = [], [], []

    with csv_file.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, dialect=dialect)

        for g, row in enumerate(reader, 1):
            try:
                winner = int(row[winner_col])
                moves_str = row[moves_col].strip()
            except (KeyError, ValueError):
                print(f"✗ Bad row {g}, skipped", file=sys.stderr)
                continue

            board, colour = start_board(), "B"
            moves = [moves_str[i : i + 2] for i in range(0, len(moves_str), 2)]

            for ply, mv in enumerate(moves, 1):
                try:
                    r, c = token_rc(mv)
                except ValueError:
                    print(f"✗ Illegal token {mv!r} (game {g}, ply {ply})", file=sys.stderr)
                    continue

                fl = flips(board, r, c, colour)
                if not fl:
                    print(f"✗ Illegal move {mv!r} (game {g}, ply {ply})", file=sys.stderr)
                    continue

                states.append("".join(board.flatten()))
                policy = np.zeros(64, np.float32)
                policy[r * 8 + c] = 1.0
                policies.append(policy)
                values.append(winner if colour == "B" else -winner)

                board[r, c] = colour
                for i, j in fl:
                    board[i, j] = colour
                colour = "W" if colour == "B" else "B"

    np.savez_compressed(
        "game_data_board_size8_othello.npz",
        states=np.array(states, dtype=object),
        policies=np.array(policies, dtype=np.float32),
        values=np.array(values, dtype=np.int8),
    )
    print(f" wrote {len(states):,} positions → game_data_board_size8_othello.npz")



def choose_dialect(csv_path: Path) -> csv.Dialect:
    """Try csv.Sniffer(); fall back to comma if it fails."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            return csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        except csv.Error:
            return csv.get_dialect("excel")  # ',' delimiter


# ── CLI wrapper (notebook-safe) ────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Othello CSV → .npz", allow_abbrev=False)
    ap.add_argument("csv_file", type=Path, help="CSV with winner & game_moves columns")
    args, _unknown = ap.parse_known_args()  # ignore Jupyter’s hidden -f argument

    if not args.csv_file.exists():
        sys.exit(f"No such file: {args.csv_file}")

    # 1️⃣ pick a dialect (delimiter)
