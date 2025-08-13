import os, zipfile, pathlib
import random

board = [["1", "2", "3"],["4", "5", "6"],["7", "8", "9"]]


# D0	identity
# D1	rotate 90°
# D2	rotate 180°
# D3	rotate 270°
# D4	mirror (vertical flip)
# D5	mirror + rotate 90°
# D6	mirror + rotate 180°
# D7	mirror + rotate 270°
# used to create multiple times dataset and prevent overfitting
def random_dihedral_rotation(board):
    mirroring = random.choice([True, False])
    random_angle = random.choice([0, 90,180,270])
    result = board
    if mirroring:
        result = mirror_board(result)
    result = rotate_board(random_angle, result)
    return result

# script to check valid keras file

def is_valid_keras_file(file_name):
    f = pathlib.Path(file_name)
    print("size:", f.stat().st_size, "bytes")
    print("is zip?", zipfile.is_zipfile(f))


# f = pathlib.Path("epoch06-val5.9491.keras")
# print("size:", f.stat().st_size, "bytes")
# print("is zip?", zipfile.is_zipfile(f))


def display_board(board):
    for row in board:
        print(row)

# helper method to rotate a board by 90 degrees
def rotate_90(board):
    size = len(board)
    result = [["." for value in range(size)] for number in range(size)]
    for row in range(size):
        for column in range(size):
            result[column][size - 1 - row] = board[row][column]
    return result

# helper method to rotate by 90 multiple by doing 90 rotations repeatedly
def rotate_board(angle, board):
    size = len(board)
    multiple = int(angle/90)
    if multiple % 4 == 0:
        return board
    for item in range(multiple):
        board = rotate_90(board)
    return board

def mirror_board(board):
    size = len(board)
    result = [["." for value in range(size)] for number in range(size)]
    for index in range(size):
        result[index] = list(reversed(board[index]))
    return result

