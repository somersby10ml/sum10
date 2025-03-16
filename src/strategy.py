import json
from typing import List

import numpy as np
from numba import njit

# 상수 정의
HEIGHT = 10
WIDTH = 17
MAX_NUM_MOVES = HEIGHT * WIDTH // 2
D = 20

@njit
def hash_grid(grid):
    """그리드에 대한 해시값을 계산합니다."""
    hash_val = 0
    for row in grid:
        for value in row:
            hash_val = (hash_val * 11 + int(value)) & 0xFFFFFFFFFFFFFFFF
    return hash_val

@njit
def create_new_grid(grid, x, y, width, height):
    """영역이 제거된 새 격자를 생성합니다."""
    new_grid = grid.copy()
    for xx in range(x, x + width):
        for yy in range(y, y + height):
            new_grid[yy, xx] = 0
    return new_grid

@njit
def count_non_zero_cells(grid, x, y, width, height):
    """영역 내 0이 아닌 셀 수를 계산합니다."""
    count = 0
    for xx in range(x, x + width):
        for yy in range(y, y + height):
            if grid[yy, xx] > 0:
                count += 1
    return count

@njit
def calculate_cumulative_sums(grid):
    """각 행에 대한 누적합을 계산합니다."""
    cum_sums = np.zeros((HEIGHT, WIDTH + 1), dtype=np.uint32)
    for i in range(HEIGHT):
        for j in range(1, WIDTH + 1):
            cum_sums[i, j] = cum_sums[i, j - 1] + grid[i, j - 1]
    return cum_sums

@njit
def update_best_moves(moves, has_move, x, y, width, height, count):
    """최고의 D개 이동 목록을 업데이트합니다."""
    for i in range(D):
        if not has_move[i] or moves[i][4] > count:
            # 공간 만들기 - 슬라이싱으로 간소화
            if i < D - 1:
                moves[i+1:] = moves[i:D-1]
                has_move[i+1:] = has_move[i:D-1]
            
            moves[i] = (x, y, width, height, count)
            has_move[i] = True
            break
    return moves, has_move

@njit
def calculate_sum_for_area(cum_sums, x, y, width, height):
    """지정된 영역의 합계를 계산합니다."""
    return cum_sums[y + height - 1, x + width] - cum_sums[y + height - 1, x]

@njit
def find_best_moves(grid, cum_sums):
    """최고의 D개 이동을 찾습니다."""
    # 튜플 배열로 이동 저장 (x, y, width, height, count)
    moves = [(0, 0, 0, 0, 0)] * D
    has_move = [False] * D
    
    # 가능한 모든 x, y 조합을 먼저 생성
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # 각 시작점에서 가능한 너비를 반복
            for width in range(1, WIDTH - x + 1):
                sum_val = 0
                # 각 너비에서 높이를 증가시키며 합계 계산
                for height in range(1, HEIGHT - y + 1):
                    sum_val += calculate_sum_for_area(cum_sums, x, y, width, height)
                    
                    # 합계가 10인 경우만 처리
                    if sum_val == 10:
                        count = count_non_zero_cells(grid, x, y, width, height)
                        moves, has_move = update_best_moves(moves, has_move, x, y, width, height, count)
    
    # 유효한 이동만 필터링하여 반환
    return [moves[i] for i in range(D) if has_move[i]]


def find_strategy(grid):
    """최적의 전략을 찾기 위해 재귀적으로 탐색합니다."""
    best_strategy = {"boxes": [], "score": 0}
    visited = set()
    best_intermediate_scores = [float('inf')] * MAX_NUM_MOVES
    
    def recurse(grid, current_boxes, current_score, num_moves):
        # 현재 전략이 더 좋다면 최고 전략 업데이트
        if current_score > best_strategy["score"]:
            best_strategy["boxes"] = current_boxes.copy()
            best_strategy["score"] = current_score
        
        # 중간 점수 기반 가지치기
        if current_score < best_intermediate_scores[num_moves]:
            best_intermediate_scores[num_moves] = current_score
        
        if current_score > best_intermediate_scores[num_moves] + 5:
            return
        
        # 이미 방문한 상태 확인
        grid_hash = hash_grid(grid)
        if grid_hash in visited:
            return
        
        visited.add(grid_hash)
        
        # 탐색 공간 제한
        if len(visited) > 100000:
            return
        
        # 누적합 계산
        cum_sums = calculate_cumulative_sums(grid)
        
        # 최상의 이동 찾기
        moves = find_best_moves(grid, cum_sums)
        
        # 각 이동 시도
        for x, y, width, height, count in moves:
            # 새 격자 생성
            new_grid = create_new_grid(grid, x, y, width, height)
            
            # 전략 업데이트 및 재귀 호출
            current_boxes.append((x, y, width, height))
            recurse(new_grid, current_boxes, current_score + count, num_moves + 1)
            current_boxes.pop()  # 백트래킹
    
    # 재귀 탐색 시작
    recurse(grid, [], 0, 0)
    
    # Box 객체로 변환
    result_boxes = []
    for x, y, width, height in best_strategy["boxes"]:
        result_boxes.append(Box(x, y, width, height))
    
    return Strategy(result_boxes, best_strategy["score"])


class Box:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def __repr__(self):
        return f"Box(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
    
    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
        


class Strategy:
    def __init__(self, boxes: List[Box] = None, score: int = 0):
        self.boxes = boxes if boxes is not None else []
        self.score = score
    
    def __repr__(self):
        return f"Strategy(boxes={self.boxes}, score={self.score})"
    
    def to_dict(self):
        return {
            "boxes": [box.to_dict() for box in self.boxes],
            "score": self.score
        }
        
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Box):
            return obj.to_dict()
        elif isinstance(obj, Strategy):
            return obj.to_dict()
        return super().default(obj)
    
