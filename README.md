# Othello AI Arena – AlphaZero-Inspired Game AI

A full-stack **Othello AI platform** featuring **Alpha-Beta**, **Monte Carlo Tree Search (MCTS)**, and **Neural-MCTS (AlphaZero-style)** agents.  
Powered by a **deep residual CNN** for policy & value prediction, with **self-play reinforcement learning** and **research-grade performance optimizations**.

---

## Features

- **Multi-Agent AI**  
  - Alpha-Beta pruning with **iterative deepening & aspiration window search**  
  - Pure MCTS and Neural-MCTS guided by policy/value network  
  - Flexible **human-vs-AI, AI-vs-AI, and self-play** modes

- **Neural Network**  
  - Residual CNN trained on **150K+ self-play and expert games**  
  - Dual-head output: **policy logits** (move probabilities) & **value prediction** (win/draw/loss)  
  - **XLA JIT-compiled inference** for lightning-fast batch predictions

- **Performance Engineering**  
  - **Transposition tables** for Alpha-Beta and MCTS caching  
  - **Iterative deepening** with aspiration window for deep search  
  - **Temperature-based softmax move selection** for exploration/exploitation control  

---

## Performance Benchmarks

| Scenario                                     | Before Optimization | After Optimization | Speedup  |
|---------------------------------------------|--------------------:|------------------:|--------:|
| **5×5 Tic-Tac-Toe Alpha-Beta (First Move)**  | 23.4 s              | 44 ms             | **~500×** |
| **4×4 Neural-MCTS (1,500 sims)**             | 127 s               | 8 s               | **~16×** |

**Key Optimizations:**  
- Transposition tables for repeated state evaluation  
- XLA fast inference for MCTS neural predictions  
- Iterative deepening + aspiration windows for selective search

---

## Tech Stack

- **Languages:** Python  
- **AI/ML:** TensorFlow (Keras), NumPy, SciKit-Learn  
- **Algorithms:** Alpha-Beta, MCTS, Neural-MCTS  
- **Optimizations:** XLA JIT, Transposition Tables, Aspiration Window  
- *(Planned)* Flask + WebSockets for real-time web gameplay with PostgreSQL state persistence

---

## Project Structure

```
Othello-AI-Arena/
│── boardgame.py           # Base game class
│── tictactoe_variant.py   # Generalized Tic-Tac-Toe implementation
│── othello.py             # Othello game logic
│── Mcts.py                # Monte Carlo Tree Search
│── Node.py                # Tree node for MCTS
│── Neural_Net.py          # Residual CNN for policy & value prediction
│── Neural_Net_Utils.py    # Data prep & model training utilities
│── self_play_bot.py       # Self-play and dataset generation
│── common_utils.py        # Helper functions & hashing
│── utility_methods.py     # Game setup & CLI helpers
│── constant_strings.py    # Global constants
│── game_data_*.npz        # Training datasets (generated)
│── LICENSE                # MIT License
└── README.md              # Project documentation
```

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/Nairvishwa24998/Othello-AI-Arena.git
cd Othello-AI-Arena

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install requirements
pip install -r requirements.txt

# Run self-play or AI match simulation
python self_play_bot.py
```

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

### Notes

- This project is **ongoing**, with **Flask + WebSockets real-time gameplay** under active development.  
- Current focus: **Neural-MCTS pipeline**, **self-play data generation**, and **scalable search optimizations**.

---
