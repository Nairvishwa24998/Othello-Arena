import matplotlib.pyplot as plt
import numpy as np
import random


def generate_random_othello_board():
    # 0: empty, 1: black, 2: white
    return np.random.choice([0, 1, 2], size=(8, 8), p=[0.5, 0.25, 0.25])


def draw_othello_board(board, filename="othello_board.png"):
    board_color = '#00896B'
    # Create figure with green background
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=board_color)

    # Set axis properties
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('forestgreen')  # Set axis background to green

    # Draw grid lines
    for i in range(9):
        ax.plot([i, i], [0, 8], color='black', linewidth=2)
        ax.plot([0, 8], [i, i], color='black', linewidth=2)

    # Draw pieces
    for row in range(8):
        for col in range(8):
            center_x = col + 0.5
            center_y = 7.5 - row  # Flip vertically for standard orientation

            if board[row, col] == 1:  # black piece
                piece = plt.Circle((center_x, center_y), 0.4, color='black', zorder=10)
                ax.add_patch(piece)
            elif board[row, col] == 2:  # white piece
                piece = plt.Circle((center_x, center_y), 0.4, color='white', zorder=10)
                ax.add_patch(piece)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Othello board saved as '{filename}'")


# Generate and draw
board = generate_random_othello_board()
draw_othello_board(board)
