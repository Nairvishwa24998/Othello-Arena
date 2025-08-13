
"""
AlphaZero-style self-play → train loop for Othello (8x8) with:
- Root Dirichlet noise (diverse self-play)
- Disk-backed replay using sharded NPZ files (keeps RAM low on Colab)

Usage:
  python az_othello_loop.py

Notes:
- Expects your repo modules (Neural_Net, Neural_Net_Utils, othello, etc.) to be importable.
- Saves/loads the "best" network where your MCTS+NN already looks:
    weights_othello_8/othello-8.keras
- No arena gating yet (single global net).
"""

from pathlib import Path
import random
import time
from typing import List, Tuple
import os
import numpy as np
import tensorflow as tf

from Neural_Net import Neural_Net
from Neural_Net_Utils import flattened_board_to_tensor
from constant_strings import GAME_OTHELLO, OTHELLO_BOARD_SIZE, MCTS_NN
try:
    from constant_strings import MCTS
    _ALLOWED_AI_TYPES = (MCTS_NN, MCTS)
except Exception:
    _ALLOWED_AI_TYPES = (MCTS_NN,)

from othello import Othello


# ==================== Dirichlet noise (root) ====================
def enable_dirichlet_noise_for_othello(
    epsilon: float = 0.25,       # AlphaZero used ~0.25
    alpha: float = 0.30,         # Dirichlet concentration (tune per game)
    noisy_moves_until: int = 12, # add noise only in the opening
    store_noisy_policy: bool = True
):
    """Monkey-patch Othello.make_ai_move to add Dirichlet noise to root policy."""
    if not hasattr(Othello, "_orig_make_ai_move"):
        Othello._orig_make_ai_move = Othello.make_ai_move
    if not hasattr(Othello, "_orig_select_optimal_ai_move_mcts"):
        Othello._orig_select_optimal_ai_move_mcts = Othello.select_optimal_ai_move_mcts

    def _noisy_make_ai_move(self, ai_type):
        if ai_type not in _ALLOWED_AI_TYPES:
            return Othello._orig_make_ai_move(self, ai_type)

        ai_player_code = self.current_player()
        opponent_code = 1 - ai_player_code
        ai_symbol = self.get_player_symbol(ai_player_code)
        opp_symbol = self.get_player_symbol(opponent_code)
        possible_moves = self.get_possible_moves(ai_player_code)

        pre_move_flattened_state_2d = "".join(str(cell) for row in self.board for cell in row)

        conclusive = self.check_immediate_result(possible_moves)
        if conclusive is not None:
            r, c = conclusive
            policy_out = np.zeros(self.size * self.size, dtype=np.float32)
            policy_out[r * self.size + c] = 1.0
        else:
            greedy_move, policy_from_mcts = Othello._orig_select_optimal_ai_move_mcts(self)

            legal_idx: List[int] = [rr * self.size + cc for (rr, cc) in possible_moves]
            p = policy_from_mcts[legal_idx].astype(np.float32)
            s = float(p.sum())
            if s <= 0.0:
                p = np.ones(len(legal_idx), dtype=np.float32) / max(1, len(legal_idx))
            else:
                p /= s

            use_noise = (self.total_moves < noisy_moves_until)
            if use_noise:
                noise = np.random.dirichlet([alpha] * len(legal_idx)).astype(np.float32)
                p_mix = (1.0 - epsilon) * p + epsilon * noise
                p_mix /= p_mix.sum()
                choice = int(np.random.choice(len(legal_idx), p=p_mix))
                stored_policy = p_mix if store_noisy_policy else p
            else:
                choice = int(np.argmax(p))
                stored_policy = p

            flat = legal_idx[choice]
            r, c = divmod(flat, self.size)

            policy_out = np.zeros(self.size * self.size, dtype=np.float32)
            policy_out[legal_idx] = stored_policy

        # Log (state, policy) for training
        self.move_list.append((pre_move_flattened_state_2d, policy_out))

        # Apply move
        self.selective_print((r, c))
        self.board[r][c] = ai_symbol
        self.increment_total_move_count()
        if not self.simulation_mode:
            self.display_board()
            time.sleep(0.5)
        self.implement_flips(r, c, ai_symbol, opp_symbol)
        self.display_board()
        self.last_moved = ai_player_code
        print(f"AI played {(r, c)}")

    Othello.make_ai_move = _noisy_make_ai_move
# ===============================================================


def weights_dir(game: str, size: int) -> Path:
    return Path(f"weights_{game}_{size}")


def weights_path(game: str, size: int) -> Path:
    return weights_dir(game, size) / f"{game}-{size}.keras"


def ensure_best_exists(game: str = GAME_OTHELLO, size: int = OTHELLO_BOARD_SIZE):
    wpath = weights_path(game, size)
    wpath.parent.mkdir(parents=True, exist_ok=True)
    if not wpath.exists():
        print(f"[init] No existing best model at {wpath}. Initializing a fresh network.")
        nn = Neural_Net(game=game, size=size)
        nn.save(str(wpath))
    else:
        print(f"[ok] Found existing best model at {wpath}.")


# ==================== Disk-backed replay (NPZ shards) ====================
class NPZShardWriter:
    """Buffers positions and flushes to compressed .npz shards to cap RAM."""
    def __init__(self, out_dir: Path, shard_size: int = 20000, prefix: str = "othello8"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self._S: List[np.ndarray] = []
        self._P: List[np.ndarray] = []
        self._V: List[np.ndarray] = []
        self._shard_counter = 0
        self._paths: List[Tuple[Path, int]] = []  # (path, count)

    @property
    def paths(self) -> List[Tuple[Path, int]]:
        return list(self._paths)

    def _flush_one(self, S: List[np.ndarray], P: List[np.ndarray], V: List[np.ndarray]):
        n = len(S)
        if n == 0:
            return None, 0
        ts = time.strftime("%Y%m%dT%H%M%S")
        name = f"{self.prefix}_shard_{ts}_{self._shard_counter:04d}.npz"
        path = self.out_dir / name

        # Compact dtypes on disk; cast back on sample.
        S_arr = np.stack(S, axis=0).astype(np.uint8, copy=False)      # 0/1 planes fit in uint8
        P_arr = np.stack(P, axis=0).astype(np.float16, copy=False)    # 2x smaller
        V_arr = np.stack(V, axis=0).astype(np.int8, copy=False)       # -1,0,1

        np.savez_compressed(path, S=S_arr, P=P_arr, V=V_arr)
        self._paths.append((path, n))
        self._shard_counter += 1
        print(f"[disk] wrote shard: {path.name}  (positions={n})")
        return path, n

    def add_batch(self, states, policies, values):
        """states: list[np.array], policies: list[np.array], values: list[np.array]"""
        self._S.extend(states)
        self._P.extend(policies)
        self._V.extend(values)

        while len(self._S) >= self.shard_size:
            # cut a shard
            S = self._S[:self.shard_size]; self._S = self._S[self.shard_size:]
            P = self._P[:self.shard_size]; self._P = self._P[self.shard_size:]
            V = self._V[:self.shard_size]; self._V = self._V[self.shard_size:]
            self._flush_one(S, P, V)

    def flush(self):
        self._flush_one(self._S, self._P, self._V)
        self._S, self._P, self._V = [], [], []


class DiskReplayNPZ:
    """Keeps an index of NPZ shards and samples batches by loading one shard at a time."""
    def __init__(self, out_dir: Path, shard_size: int, capacity_positions: int, keep_loaded_batches: int = 128):
        self.writer = NPZShardWriter(out_dir=out_dir, shard_size=shard_size, prefix="othello8")
        self.capacity_positions = capacity_positions
        self._total_positions = 0
        self._shards: List[Tuple[Path, int]] = []  # (path, count)
        self._cache = None  # (S, P, V) arrays currently loaded
        self._batches_remaining = 0
        self._keep_loaded_batches = keep_loaded_batches

    def __len__(self):
        return int(self._total_positions)

    def _refresh_index_from_writer(self):
        for path, n in self.writer.paths[len(self._shards):]:
            self._shards.append((path, n))
            self._total_positions += n
        self._prune_capacity_if_needed()

    def _prune_capacity_if_needed(self):
        while self._total_positions > self.capacity_positions and self._shards:
            # delete oldest shard on disk
            old_path, n = self._shards.pop(0)
            try:
                os.remove(old_path)
                print(f"[disk] removed old shard to honor capacity: {old_path.name}")
            except OSError:
                pass
            self._total_positions -= n

    def add_batch(self, states, policies, values):
        self.writer.add_batch(states, policies, values)
        self._refresh_index_from_writer()

    def flush(self):
        self.writer.flush()
        self._refresh_index_from_writer()

    # ------- sampling -------
    def _load_random_shard(self):
        if not self._shards:
            raise RuntimeError("No shards available to sample from.")
        path, n = random.choice(self._shards)
        with np.load(path) as z:
            S = z["S"]     # uint8 [N,H,W,C]
            P = z["P"]     # float16 [N,64]
            V = z["V"]     # int8 [N,1]
            # Load into memory just this shard
            S = S.astype(np.float32)
            P = P.astype(np.float32)
            V = V.astype(np.float32)
        self._cache = (S, P, V)
        self._batches_remaining = self._keep_loaded_batches

    def sample(self, batch_size: int):
        if self._cache is None or self._batches_remaining <= 0:
            self._load_random_shard()
        S, P, V = self._cache
        n = S.shape[0]
        idx = np.random.randint(0, n, size=(batch_size,))
        self._batches_remaining -= 1
        return S[idx], {"policy_logits": P[idx], "value": V[idx]}
# ===========================================================================


def play_one_game_collect(game_size: int = 8):
    game = Othello(size=game_size, vs_human=False, ai_player_code=0, ai_type=MCTS_NN, simulation_mode=True)
    while True:
        turn = game.current_player()
        if turn == -1:
            break
        game.ai_player_code = turn
        game.set_AI_type(MCTS_NN)
        game.make_ai_move(MCTS_NN)

    result = game.detect_win_loss()  # 1 (B wins), -1 (W wins), 0 draw

    states, policies, values = [], [], []
    for idx, (flat_state, policy) in enumerate(game.move_list):
        tensor = flattened_board_to_tensor(flat_state, GAME_OTHELLO)
        # store compact; we'll cast back in sampler
        states.append(tensor.astype(np.uint8))       # 0/1 planes → uint8
        policies.append(policy.astype(np.float16))   # logits ok in fp16
        eff = result if (idx % 2 == 0) else -result
        values.append(np.array([eff], dtype=np.int8))
    return states, policies, values, result


def generate_selfplay_positions_to_disk(
    replay: DiskReplayNPZ,
    target_positions: int,
    game_size: int = 8,
    report_every_games: int = 5
):
    total_positions = 0
    games = 0
    w = l = d = 0

    while total_positions < target_positions:
        S, P, V, res = play_one_game_collect(game_size)
        total_positions += len(S)
        games += 1
        if res == 1: w += 1
        elif res == -1: l += 1
        else: d += 1

        replay.add_batch(S, P, V)

        if games % report_every_games == 0:
            print(f"[selfplay] games={games} positions={total_positions} (W:{w} L:{l} D:{d})")

    replay.flush()
    return total_positions, games, (w, l, d)


def train_candidate_from_replay(best_model_path: Path, replay: DiskReplayNPZ,
                                steps: int = 40_000, batch_size: int = 256,
                                policy_w: float = 1.0, value_w: float = 1.5,
                                base_lr: float = 1e-3):
    nn = Neural_Net(game=GAME_OTHELLO, size=OTHELLO_BOARD_SIZE)
    nn.load(str(best_model_path))

    steps_per_epoch = max(1, steps // batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr, decay_steps=steps_per_epoch
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    nn.model.compile(
        optimizer=opt,
        loss={
            "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            "value": tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={"policy_logits": policy_w, "value": value_w},
    )

    def data_gen():
        while True:
            yield replay.sample(batch_size)

    print(f"[train] steps={steps} batch_size={batch_size} (~{steps//batch_size} epochs)")
    nn.model.fit(
        data_gen(),
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        verbose=1,
    )
    return nn.model


def replace_best(model, game: str = GAME_OTHELLO, size: int = OTHELLO_BOARD_SIZE):
    path = weights_path(game, size)
    model.save(str(path))
    print(f"[save] Updated best weights at: {path}")


def run_training_loop(
    num_iters: int = 5,
    target_positions: int = 60_000,
    replay_capacity: int = 1_000_000,   # positions across shards
    shard_size: int = 20000,            # positions per .npz shard
    batch_size: int = 256,
    steps: int = 40_000,
    policy_w: float = 1.0,
    value_w: float = 1.5,
    base_lr: float = 1e-3,
):
    ensure_best_exists(GAME_OTHELLO, OTHELLO_BOARD_SIZE)
    best_path = weights_path(GAME_OTHELLO, OTHELLO_BOARD_SIZE)

    # disk-backed replay under ./selfplay_npz/
    replay = DiskReplayNPZ(
        out_dir=Path("selfplay_npz"),
        shard_size=shard_size,
        capacity_positions=replay_capacity,
        keep_loaded_batches=128,   # number of batches to draw from a loaded shard before rotating
    )

    for it in range(1, num_iters + 1):
        print(f"\n========== Iteration {it}/{num_iters} ==========")
        print(f"[selfplay] Generating ~{target_positions} positions → NPZ shards…")
        pos, games, (w, l, d) = generate_selfplay_positions_to_disk(
            replay=replay,
            target_positions=target_positions,
            game_size=OTHELLO_BOARD_SIZE,
            report_every_games=5
        )
        print(f"[selfplay] done: games={games}, positions={pos} (W:{w} L:{l} D:{d})")
        print(f"[replay] on-disk positions (capped) ~ {len(replay)}")

        candidate = train_candidate_from_replay(
            best_model_path=best_path,
            replay=replay,
            steps=steps,
            batch_size=batch_size,
            policy_w=policy_w,
            value_w=value_w,
            base_lr=base_lr,
        )

        replace_best(candidate, GAME_OTHELLO, OTHELLO_BOARD_SIZE)

    print("\n[done] Training loop completed.")


if __name__ == "__main__":
    # Enable Dirichlet noise for varied, high-quality self-play data
    enable_dirichlet_noise_for_othello(
        epsilon=0.25,
        alpha=0.30,
        noisy_moves_until=12,
        store_noisy_policy=True
    )

    # Start small to sanity-check the flow, then scale up.
    run_training_loop(
        num_iters=2,
        target_positions=10_000,    # try 60_000+ when you're ready
        replay_capacity=300_000,    # cap across shards (positions)
        shard_size=20_000,          # ~10–30k is a nice shard size
        batch_size=256,
        steps=10_000,               # try 40_000+ for longer training
        policy_w=1.0,
        value_w=1.5,
        base_lr=1e-3,
    )
