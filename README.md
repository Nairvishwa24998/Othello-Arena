# Othello AI Arena – AlphaZero-Inspired Game AI

A **Othello AI platform** featuring **Alpha-Beta**, **Monte Carlo Tree Search (MCTS)**, and **Neural‑MCTS (AlphaZero‑style)** agents.  
Powered by a **deep residual CNN** for policy & value prediction, with **self-play reinforcement learning** and **research‑grade performance optimizations**.

---

## Features

- **Multi‑Agent AI**
  - Alpha‑Beta pruning with **iterative deepening & aspiration window search**
  - Pure MCTS and Neural‑MCTS guided by a policy/value network
  - Flexible **human‑vs‑AI, AI‑vs‑AI, and self‑play** modes

- **Neural Network**
  - Residual CNN trained on **self‑play and expert games**
  - Dual‑head outputs: **policy logits** (move probabilities) & **value** (win/draw/loss)
  - **XLA JIT‑compiled** inference for fast predictions

- **Performance Engineering**
  - **Transposition tables** for Alpha‑Beta and MCTS caching
  - **Iterative deepening** with aspiration windows
  - **Temperature‑based softmax** move selection for exploration/exploitation

---

## Performance Benchmarks

| Scenario                                     | Before Optimization | After Optimization | Speedup  |
|---------------------------------------------|--------------------:|------------------:|--------:|
| **5×5 Tic‑Tac‑Toe Alpha‑Beta (First Move)** | 23.4 s              | 44 ms             | **~500×** |
| **8×8 Othello Neural‑MCTS (3000 sims)**     | 127 s               | 8 s               | **~16×**  |

**Key Optimizations**
- Transposition tables for repeated state evaluation  
- XLA fast inference for MCTS neural predictions  
- Iterative deepening + aspiration windows for selective search

---

## Tech Stack

- **Language:** Python  
- **AI/ML:** TensorFlow (Keras), NumPy, SciKit‑Learn, SciPy  
- **Algorithms:** Alpha‑Beta, MCTS, Neural‑MCTS  
- **Optimizations:** XLA JIT, Transposition Tables, Aspiration Window  
- *(Planned)* Flask + WebSockets for real‑time web gameplay with PostgreSQL state persistence

---

## Project Structure

```
Othello-AI-Arena/
├── accuracy_testing/
│   ├── othello_accuracy_testing_script.py
│   └── tictactoe_accuracy_testing_script.py
├── checkpoints/
├── constant_strings.py
├── data_gen_and_cleaning/
│   ├── data_cleaning_othello_e_othello_games.py
│   ├── data_set_generation_othello.py
│   └── othello_data_augmentation.py
├── game_data_board_size3_ttt.npz
├── game_data_board_size4_ttt.npz
├── game_data_board_size5_ttt.npz
├── game_data_board_size8_othello.npz
├── game_logic_layer/
│   ├── boardgame.py
│   ├── othello.py
│   └── tictactoe_variant.py
├── neural_net.py
├── neural_net_utils.py
├── requirements.txt
├── search_layer/
│   ├── Node.py
│   ├── mcts_othello.py
│   ├── mcts_parent.py
│   ├── mcts_ttt.py
│   ├── othello_ab_pruning_helper.py
│   └── ttt_ab_pruning_helper.py
├── static/
│   └── logo.png
├── training_logs/
│   └── d4_90_24_training_logs/
├── user_interface/
│   └── othello_ui.py
├── utils/
│   ├── common_utils.py
│   ├── game_play_utility_methods.py
│   └── self_play_bot.py
├── weights_othello_8/
│   └── othello-8.keras
└── weights_ttt_4/
    └── ttt-4.keras
e
└── README.md                # Project documentation
```

> **Entry point:** `utility_methods.py` contains:
> ```python
> if __name__ == "__main__":
>     commence_game_play()
> ```

---

## Setup & Usage

### 🔧 Setup

1) **Clone**
```bash
git clone https://github.com/Nairvishwa24998/Othello-Arena.git
cd Othello-Arena
```

2) **(Recommended) Create & activate a virtual environment**
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3) **Install dependencies**
- If you have `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- If not, install the core deps directly:
  ```bash
  pip install numpy tensorflow keras scikit-learn scipy
  ```
  *(Optional) Create a requirements file for future installs*
  ```bash
  pip freeze > requirements.txt
  ```

4) **(Optional) Pre-trained weights for Neural‑MCTS**
- Place weights at: `weights_<game>_<size>/<game>-<size>.keras`  
  Examples:
  - `weights_othello_8/othello-8.keras`
  - `weights_tictactoe_4/tictactoe-4.keras`
- If you don’t have weights yet, you can still play with **Alpha‑Beta** or **pure MCTS**. Train later if you want Neural‑MCTS.

---

###  Usage

#### 1) Start the interactive game (calls `commence_game_play()`)

Run:
```bash
python utility_methods.py
```

The CLI will prompt you to choose:
- Game: **TicTacToe** or **Othello**
- Fresh game or **custom position**
- Opponent: **Human** or **AI**
- If AI: **play order** (AI first/second)
- If AI: **engine**
  - `0 = ALPHA_BETA_PRUNING`
  - `1 = MCTS`
  - `2 = MCTS + Neural Network` *(requires matching weights as above)* 

Extras:
- **TicTacToe:** choose board size (2–7). MTCS + NN in tictactoe only suupported for size 4*4 as of now 
- **Othello:** fixed 8×8; for custom positions, input 64 comma‑separated `B/W/.` entries.

---

#### 2) (Optional) Generate self‑play datasets

Creates `.npz` training data via AI‑vs‑AI simulations.
```bash
python self_play_bot.py
```
Inside `self_play_bot.py`, enable or tweak:
- `run_simulations(size, ai_type)` for dataset generation
- `ttt_run_bot_v_bot_matches(...)` or `othello_run_bot_v_bot_matches(...)` for AI benchmarks

---

#### 3) (Optional) Train the neural network

```bash
python Neural_Net_Utils.py
```
What it does:
- Loads datasets like `game_data_board_size4_ttt.npz`
- Splits into train/val/test
- Trains the residual CNN with policy/value heads
- Saves a model (e.g., `final_model_othello_8x8.keras`)

> For gameplay inference within MCTS‑NN, expected path is:
> `weights_<game>_<size>/<game>-<size>.keras`

---

## License

This project is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2025 Viswanath B Nair

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---
