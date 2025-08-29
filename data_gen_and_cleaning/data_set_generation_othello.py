
import os
import sys
import glob
import struct
import argparse
from typing import List, Tuple

import numpy as np

EMPTY, BLACK, WHITE = '.', 'B', 'W'
DIRS = [(-1,-1),(-1,0),(-1,1),
        ( 0,-1),        ( 0, 1),
        ( 1,-1),( 1,0),( 1, 1)]

HEADER_LEN = 16
RECORD_LEN_8x8 = 68  # per game record (after header) for 8x8 files

def opp(side: str) -> str:
    return BLACK if side == WHITE else WHITE

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8

def legal_dirs(board: List[List[str]], side: str, r: int, c: int) -> List[Tuple[int, int]]:
    if board[r][c] != EMPTY:
        return []
    other = opp(side)
    found = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        seen_other = False
        while in_bounds(rr, cc) and board[rr][cc] == other:
            rr += dr; cc += dc; seen_other = True
        if seen_other and in_bounds(rr, cc) and board[rr][cc] == side:
            found.append((dr, dc))
    return found

def has_legal(board: List[List[str]], side: str) -> bool:
    for r in range(8):
        for c in range(8):
            if legal_dirs(board, side, r, c):
                return True
    return False

def play_move(board: List[List[str]], side: str, r: int, c: int) -> List[List[str]]:
    b2 = [row[:] for row in board]
    flips = legal_dirs(b2, side, r, c)
    b2[r][c] = side
    for dr, dc in flips:
        rr, cc = r + dr, c + dc
        while b2[rr][cc] == opp(side):
            b2[rr][cc] = side
            rr += dr; cc += dc
    return b2

def start_board() -> List[List[str]]:
    b = [[EMPTY]*8 for _ in range(8)]
    # Standard start: d4=W, e4=B, d5=B, e5=W
    b[3][3] = WHITE; b[3][4] = BLACK
    b[4][3] = BLACK; b[4][4] = WHITE
    return b

def board_to_str(board: List[List[str]]) -> str:
    return "".join(board[r][c] for r in range(8) for c in range(8))

# WTHOR parsing (8x8)

def parse_wtb_record(rec: bytes):
    tourn, pid_b, pid_w, black_real, theo = struct.unpack_from("<HHHBB", rec, 0)
    moves_raw = rec[8:]  # 60 bytes
    codes = []
    for m in moves_raw:
        if m == 0:
            break
        row, col = divmod(m, 10)
        if 1 <= row <= 8 and 1 <= col <= 8:
            codes.append(m)
        else:
            return tourn, pid_b, pid_w, black_real, []  # invalid code → unusable game
    return tourn, pid_b, pid_w, black_real, codes

def decode_code_to_rc_idx(code: int) -> Tuple[int, int, int]:
    row, col = divmod(code, 10)  # row,col in 1..8
    r, c = row - 1, col - 1
    return r, c, r*8 + c

# -----------------------------
# Convert one game → samples
# -----------------------------
def game_to_samples(black_final_score: int, move_codes: List[int]):
    """Emit (states, policies, values) for each REAL move (no row for passes)."""
    if not move_codes:
        return [], [], [], []

    # Final result from BLACK POV
    if black_final_score > 32:  res_black = +1
    elif black_final_score < 32: res_black = -1
    else:                        res_black = 0

    board = start_board()
    side  = BLACK
    states, policies, values, to_move = [], [], [], []

    for code in move_codes:
        # Handle pass if needed
        if not has_legal(board, side):
            side = opp(side)
            if not has_legal(board, side):  # neither can move → over
                break

        r, c, idx = decode_code_to_rc_idx(code)

        # If still illegal, try one implicit pass (WTHOR doesn't record passes)
        if not legal_dirs(board, side, r, c):
            if has_legal(board, opp(side)):
                side = opp(side)
            if not legal_dirs(board, side, r, c):
                return [], [], [], []  # corrupt

        # Emit BEFORE move
        states.append(board_to_str(board))
        onehot = np.zeros(64, dtype=np.float32); onehot[idx] = 1.0
        policies.append(onehot)

        # Side-to-move label
        v = res_black if side == BLACK else -res_black
        to_move.append(np.int8(1 if side == BLACK else 0))
        values.append(np.int8(v))

        # Play + switch
        board = play_move(board, side, r, c)
        side  = opp(side)

    return states, policies, values, to_move

def verify_game(codes: List[int], expected_black_score: int) -> bool:
    """Re-simulate and check final Black disc count."""
    board = start_board()
    side  = BLACK
    for code in codes:
        if not has_legal(board, side):
            side = opp(side)
            if not has_legal(board, side):
                break
        r, c, _ = decode_code_to_rc_idx(code)
        if not legal_dirs(board, side, r, c):
            if has_legal(board, opp(side)):
                side = opp(side)
            if not legal_dirs(board, side, r, c):
                return False
        board = play_move(board, side, r, c)
        side  = opp(side)
    final_black = sum(ch == BLACK for ch in board_to_str(board))
    return final_black == expected_black_score

# -----------------------------
# File discovery (pick ONE file per year)
# -----------------------------
def find_wtb_files(root: str, years: range) -> List[str]:
    """Pick exactly one .wtb/.WTB per year; prefer .wtb; then shortest path."""
    picked = []
    for y in years:
        matches = []
        for pat in (
            os.path.join(root, f"WTH_{y}.wtb"),
            os.path.join(root, f"WTH_{y}.WTB"),
            os.path.join(root, "**", f"WTH_{y}.wtb"),
            os.path.join(root, "**", f"WTH_{y}.WTB"),
        ):
            matches.extend(glob.glob(pat, recursive=True))
        if not matches:
            continue
        # de-dupe paths (case-insensitive on Windows)
        uniq = {}
        for p in matches:
            uniq[os.path.normcase(os.path.abspath(p))] = p
        picks = list(uniq.values())
        def score(p):
            return (0 if p.lower().endswith(".wtb") else 1, len(p), p.lower())
        picks.sort(key=score)
        picked.append(picks[0])
    return picked

# -----------------------------
# Main builder
# -----------------------------
def build_npz_from_wthor(
    input_dir: str,
    years = range(1990, 2025),
    out_path = "game_data_board_size8_othello_STM.npz",
    strict_verify: bool = True,
    compress: bool = False
):
    files = find_wtb_files(input_dir, years)
    if not files:
        print(f"[ERROR] No WTH_YYYY.wtb/.WTB files found in: {input_dir}")
        sys.exit(2)

    print("Using files:")
    for p in files:
        print(" -", p)

    all_states: List[str] = []
    all_policies: List[np.ndarray] = []
    all_values: List[int] = []
    all_to_move: List[int] = []

    seen_games = set()   # (tourn, pid_b, pid_w, black_real, bytes(codes))
    dup_games = 0

    total_games = kept_games = skipped_games = 0

    for fpath in files:
        data = open(fpath, "rb").read()
        if len(data) < HEADER_LEN or (len(data) - HEADER_LEN) % RECORD_LEN_8x8 != 0:
            print(f"[WARN] {os.path.basename(fpath)} has unexpected size; skipping.")
            continue

        body = data[HEADER_LEN:]
        n_records = len(body) // RECORD_LEN_8x8
        file_kept = file_skipped = 0

        for i in range(n_records):
            total_games += 1
            rec = body[i*RECORD_LEN_8x8:(i+1)*RECORD_LEN_8x8]
            tourn, pid_b, pid_w, black_real, codes = parse_wtb_record(rec)
            if not codes:
                file_skipped += 1
                continue

            # duplicate game guard (across files)
            game_key = (tourn, pid_b, pid_w, black_real, bytes(codes))
            if game_key in seen_games:
                dup_games += 1
                file_skipped += 1
                continue
            seen_games.add(game_key)

            if strict_verify and not verify_game(codes, black_real):
                file_skipped += 1
                continue

            states, policies, values, to_move_list = game_to_samples(black_real, codes)
            if not states:
                file_skipped += 1
                continue

            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            all_to_move.extend(to_move_list)
            kept_games += 1
            file_kept   += 1

        print(f"{os.path.basename(fpath)}: records={n_records}, kept={file_kept}, skipped={file_skipped}")
        skipped_games += file_skipped

    states_arr   = np.array(all_states, dtype=object)
    policies_arr = np.vstack(all_policies).astype(np.float32) if all_policies else np.zeros((0,64), np.float32)
    values_arr   = np.array(all_values, dtype=np.int8)
    to_move_list_arr = np.array(all_to_move, dtype=np.int8)

    print(f"TOTAL games: {total_games} | kept: {kept_games} | skipped: {skipped_games}")
    print(f"Duplicate games skipped: {dup_games}")
    print(f"Output shapes: states={states_arr.shape}, policies={policies_arr.shape}, values={values_arr.shape}")

    if compress:
        np.savez_compressed(out_path, states=states_arr, policies=policies_arr, values=values_arr, to_move_list = to_move_list_arr)
    else:
        np.savez(out_path, states=states_arr, policies=policies_arr, values=values_arr, to_move_list = to_move_list_arr)

    print(f"Saved → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert WTHOR .wtb (8x8) to STM NPZ (AlphaZero-style).")
    ap.add_argument("input_dir", help="Folder containing WTH_YYYY.wtb/.WTB (searches recursively).")
    ap.add_argument("out_path", nargs="?", default="game_data_board_size8_othello_STM.npz",
                    help="Output NPZ file path.")
    ap.add_argument("--start", type=int, default=1990, help="Start year (default 2010).")
    ap.add_argument("--end",   type=int, default=2025, help="End year inclusive (default 2024).")
    ap.add_argument("--no-verify", action="store_true", help="Disable strict final-score verification.")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed.")
    args = ap.parse_args()

    years = range(args.start, args.end + 1)
    build_npz_from_wthor(
        input_dir=args.input_dir,
        years=years,
        out_path=args.out_path,
        strict_verify=not args.no_verify,
        compress=args.compress
    )

if __name__ == "__main__":
    main()
