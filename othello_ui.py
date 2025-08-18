import pygame
from pygame.sprite import LayeredDirty

from constant_strings import UI_LINE_THICKNESS, UI_MARGIN, UI_PIXEL_COUNT, GRID, BLACK, WHITE, MOVE_B, MOVE_W, GREEN, \
    MAX_ICON_SIZE


def fetch_ui_config(n):
    pygame.init()
    CELL = UI_PIXEL_COUNT
    # MARG = UI_MARGIN
    MARG = max(UI_MARGIN, CELL - 12 + 10)
    LINE = UI_LINE_THICKNESS
    W = H = MARG * 2 + CELL * n
    # resolve path: <repo>/static/logo.png (adjust if your path differs)
    logo_path = "static/logo.png"
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Othello")
    add_logo(logo_path)
    # wood bordering and labels
    WOOD = (148, 96, 55)
    LABEL = (245, 226, 179)
    FONT = pygame.font.SysFont(None, max(12, int(MARG * 0.6)))

    pygame_config = {
        "pygame": pygame, "screen": screen, "n": n, "W": W, "H": H,
        "CELL": CELL, "MARG": MARG, "LINE": LINE,
        "GREEN": GREEN, "GRID": GRID, "BLACK": BLACK, "WHITE": WHITE,
        "MOVE_B": MOVE_B, "MOVE_W": MOVE_W,
        "WOOD": WOOD, "LABEL": LABEL, "FONT": FONT,
    }
    return pygame_config

def create_board_layout(screen, pygame_config):
    pg = pygame_config["pygame"]
    wood = pygame_config["WOOD"]
    board_margin = pygame_config["MARG"]
    cell = pygame_config["CELL"]
    n = pygame_config["n"]
    # --- wood frame & label bands (use the existing margin area) ---
    screen.fill(wood)  # full window wood
    # inset board area
    board_rect = (board_margin, board_margin, cell * n, cell * n)
    pg.draw.rect(screen, GREEN, board_rect)  # green board
    pg.draw.rect(screen, (110, 70, 38),  # darker inner bevel
                 (board_margin - 2, board_margin - 2, cell * n + 4, cell * n + 4), 4)

def add_alphabets_board_border(screen, pygame_config):
    cell = pygame_config["CELL"]
    column_label = pygame_config["LABEL"]
    font = pygame_config["FONT"]
    board_margin = pygame_config["MARG"]
    n = pygame_config["n"]
    # place alphabets across horizontally
    for c in range(n):
        ch = chr(ord('A') + c)
        surf = font.render(ch, True, column_label)
        rect = surf.get_rect(center=(board_margin + c * cell + cell // 2, board_margin // 2))
        screen.blit(surf, rect)

def add_numbers_board_border(screen, pygame_config):
    cell = pygame_config["CELL"]
    column_label = pygame_config["LABEL"]
    font = pygame_config["FONT"]
    board_margin = pygame_config["MARG"]
    n = pygame_config["n"]
    # place numbers vertically
    for r in range(n):
        ch = str(r + 1)
        surf = font.render(ch, True, column_label)
        rect = surf.get_rect(center=(board_margin // 2, board_margin + r * cell + cell // 2))
        screen.blit(surf, rect)

def add_grid_lines(screen, pygame_config):
    pg = pygame_config["pygame"]
    cell = pygame_config["CELL"]
    board_margin = pygame_config["MARG"]
    n = pygame_config["n"]
    line = pygame_config["LINE"]
    # draw grid lines
    for i in range(n + 1):
        x = board_margin + i * cell
        pg.draw.line(screen, GRID, (x, board_margin), (x, board_margin + cell * n), line)
        pg.draw.line(screen, GRID, (board_margin, board_margin + i * cell),
                     (board_margin + cell * n, board_margin + i * cell), line)


def place_discs(screen, pygame_config,board):
    pg = pygame_config["pygame"]
    n = pygame_config["n"]
    cell = pygame_config["CELL"]
    board_margin = pygame_config["MARG"]
    for r in range(n):
        for c in range(n):
            v = board[r][c]
            if v == MOVE_B or v == MOVE_W:
                cx = board_margin + c * cell + cell // 2
                cy = board_margin + r * cell + cell // 2
                color = BLACK if v == MOVE_B else WHITE
                pg.draw.circle(screen, color, (cx, cy), cell // 2 - 6)

def set_ui_config_to_board(pygame_config, board, black_score, white_score):
    screen = pygame_config["screen"]
    create_board_layout(screen, pygame_config)
    draw_scoreboard(screen, pygame_config, black_score=black_score, white_score=white_score)
    add_alphabets_board_border(screen, pygame_config)
    add_numbers_board_border(screen, pygame_config)
    add_grid_lines(screen, pygame_config)
    place_discs(screen,pygame_config,board)

def draw_scoreboard(screen, pygame_config, black_score=2, white_score=2):
    pg    = pygame_config["pygame"]
    W, H  = pygame_config["W"], pygame_config["H"]
    m     = pygame_config["MARG"]
    cell  = pygame_config["CELL"]
    BLACK = pygame_config["BLACK"]
    WHITE = pygame_config["WHITE"]
    label = pygame_config["LABEL"]
    font  = pygame_config["FONT"]

    r   = cell // 2 - 6
    pad = max(12, r // 2)

    # place in bottom margin, BELOW the A–H row
    letters_center = H - m // 2
    y = H - (r + 6)                # near bottom edge
    y = max(y, letters_center + 6) # ensure it's below the letters

    # Left: Black
    xL = max(8, m // 3) + r
    pg.draw.circle(screen, BLACK, (xL, y), r)
    tB = font.render(f"- {black_score}", True, label)
    screen.blit(tB, (xL + r + pad, y - tB.get_height() // 2))

    # Right: White
    tW = font.render(f"- {white_score}", True, label)
    cluster_w = (r * 2) + pad + tW.get_width()
    x_start   = W - max(8, m // 3) - cluster_w
    xR        = x_start + r
    pg.draw.circle(screen, WHITE, (xR, y), r)
    pg.draw.circle(screen, (40,40,40), (xR, y), r, 1)
    screen.blit(tW, (xR + r + pad, y - tW.get_height() // 2))



def add_logo(icon_path):
    try:
        icon = pygame.image.load(str(icon_path)).convert_alpha()
        # (optional) scale big images down so Windows/Linux tray looks crisp
        if icon.get_width() > MAX_ICON_SIZE or icon.get_height() > MAX_ICON_SIZE:
            icon = pygame.transform.smoothscale(icon, (MAX_ICON_SIZE, MAX_ICON_SIZE))
        pygame.display.set_icon(icon)
    except Exception as e:
        print("[icon warning]", e)

