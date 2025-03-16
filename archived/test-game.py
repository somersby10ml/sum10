import sys

import numpy as np
import pygame

# 제공된 게임 데이터 (numpy 배열)
grid = np.array([
    [2, 7, 7, 2, 6, 1, 5, 2, 2, 7, 9, 6, 2, 3, 1, 4, 2],
    [7, 8, 3, 9, 6, 6, 7, 1, 7, 6, 7, 3, 1, 9, 5, 9, 7],
    [3, 5, 9, 9, 6, 1, 7, 8, 1, 5, 8, 9, 9, 7, 4, 9, 4],
    [4, 4, 6, 8, 3, 5, 1, 8, 3, 2, 4, 4, 3, 9, 7, 9, 5],
    [3, 2, 9, 2, 1, 5, 2, 9, 2, 3, 9, 5, 7, 2, 3, 5, 9],
    [4, 7, 4, 2, 1, 4, 3, 7, 3, 5, 9, 3, 9, 2, 9, 4, 4],
    [4, 8, 3, 8, 8, 9, 9, 2, 4, 4, 3, 7, 1, 8, 3, 3, 5],
    [2, 1, 9, 2, 8, 9, 9, 1, 3, 1, 7, 8, 5, 9, 5, 2, 1],
    [5, 6, 7, 6, 3, 3, 7, 1, 6, 2, 6, 9, 6, 1, 2, 1, 8],
    [7, 8, 4, 5, 8, 9, 4, 3, 7, 8, 9, 9, 1, 1, 3, 5, 3]
])

# 그리드 크기 및 셀 크기 설정
ROWS = grid.shape[0]    # 10
COLS = grid.shape[1]    # 17
CELL_SIZE = 40          # 셀 하나의 픽셀 크기

# 좌표(축) 표기를 위한 여백(margin) 설정
MARGIN_LEFT = 40   # 왼쪽 여백 (y좌표 표기용)
MARGIN_TOP = 40    # 상단 여백 (x좌표 표기용)

GRID_WIDTH = COLS * CELL_SIZE
GRID_HEIGHT = ROWS * CELL_SIZE

# 전체 창 크기 (오른쪽, 아래쪽에 약간의 여백 추가)
WINDOW_WIDTH = MARGIN_LEFT + GRID_WIDTH + 20
WINDOW_HEIGHT = MARGIN_TOP + GRID_HEIGHT + 50  # 하단에 스코어 표시 공간 포함

score = 0

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Apple Game (Sum to 10)")
font = pygame.font.SysFont(None, 24)  # 좌표와 셀 내 숫자 표시용 폰트

# 드래그 관련 변수
dragging = False
start_cell = None
current_cell = None


def draw_grid():
    screen.fill((255, 255, 255))

    # x축 좌표 (상단) 표시
    for j in range(COLS):
        x = MARGIN_LEFT + j * CELL_SIZE + CELL_SIZE / 2
        y = MARGIN_TOP / 2
        label = font.render(str(j), True, (0, 0, 0))
        label_rect = label.get_rect(center=(x, y))
        screen.blit(label, label_rect)

    # y축 좌표 (좌측) 표시
    for i in range(ROWS):
        x = MARGIN_LEFT / 2
        y = MARGIN_TOP + i * CELL_SIZE + CELL_SIZE / 2
        label = font.render(str(i), True, (0, 0, 0))
        label_rect = label.get_rect(center=(x, y))
        screen.blit(label, label_rect)

    # 그리드 셀 그리기
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(MARGIN_LEFT + j * CELL_SIZE,
                               MARGIN_TOP + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # 값이 0이면 제거된 셀 (회색), 아니면 밝은 색
            color = (200, 200, 200) if grid[i, j] == 0 else (240, 240, 240)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            # 셀에 숫자 표시 (값이 0이면 표시하지 않음)
            if grid[i, j] != 0:
                text = font.render(str(grid[i, j]), True, (0, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

    # 드래그 중인 영역 표시
    if dragging and start_cell and current_cell:
        r1, c1 = start_cell
        r2, c2 = current_cell
        top = min(r1, r2)
        left = min(c1, c2)
        bottom = max(r1, r2)
        right = max(c1, c2)
        sel_rect = pygame.Rect(MARGIN_LEFT + left * CELL_SIZE,
                               MARGIN_TOP + top * CELL_SIZE,
                               (right - left + 1) * CELL_SIZE,
                               (bottom - top + 1) * CELL_SIZE)
        pygame.draw.rect(screen, (0, 255, 0), sel_rect, 3)

    # 스코어 표시 (창 하단)
    score_text = font.render("Score: " + str(score), True, (0, 0, 0))
    screen.blit(score_text, (MARGIN_LEFT, MARGIN_TOP + GRID_HEIGHT + 10))
    pygame.display.flip()


clock = pygame.time.Clock()

running = True
while running:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 좌클릭 시작
                pos = pygame.mouse.get_pos()
                # 그리드 영역 내 클릭인지 확인 (여백 제외)
                if (MARGIN_LEFT <= pos[0] < MARGIN_LEFT + GRID_WIDTH) and (MARGIN_TOP <= pos[1] < MARGIN_TOP + GRID_HEIGHT):
                    col = (pos[0] - MARGIN_LEFT) // CELL_SIZE
                    row = (pos[1] - MARGIN_TOP) // CELL_SIZE
                    start_cell = (row, col)
                    current_cell = (row, col)
                    dragging = True

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                pos = pygame.mouse.get_pos()
                if (MARGIN_LEFT <= pos[0] < MARGIN_LEFT + GRID_WIDTH) and (MARGIN_TOP <= pos[1] < MARGIN_TOP + GRID_HEIGHT):
                    col = (pos[0] - MARGIN_LEFT) // CELL_SIZE
                    row = (pos[1] - MARGIN_TOP) // CELL_SIZE
                    current_cell = (row, col)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and dragging:
                dragging = False
                pos = pygame.mouse.get_pos()
                if (MARGIN_LEFT <= pos[0] < MARGIN_LEFT + GRID_WIDTH) and (MARGIN_TOP <= pos[1] < MARGIN_TOP + GRID_HEIGHT):
                    col = (pos[0] - MARGIN_LEFT) // CELL_SIZE
                    row = (pos[1] - MARGIN_TOP) // CELL_SIZE
                    current_cell = (row, col)
                    # 선택 영역 계산
                    r1, c1 = start_cell
                    r2, c2 = current_cell
                    top = min(r1, r2)
                    left = min(c1, c2)
                    bottom = max(r1, r2)
                    right = max(c1, c2)
                    region = grid[top:bottom+1, left:right+1]
                    region_sum = np.sum(region)
                    # 합이 10이면 해당 영역 제거 및 스코어 증가
                    if region_sum == 10:
                        grid[top:bottom+1, left:right+1] = 0
                        score += 1
                start_cell = None
                current_cell = None

    draw_grid()

pygame.quit()
sys.exit()
