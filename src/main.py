import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from apple_ocr.models import (extract_digit_cells, predict_digits,
                              process_cells_to_tensors)

app = Flask(__name__)
CORS(app)

from strategy import CustomJSONEncoder, find_strategy


def setup_flask_json_encoder(app):
    app.json_encoder = CustomJSONEncoder
    
def process_image(image_bytes):
    """이미지 바이트를 처리하여 숫자 인식 결과 반환"""
    # 바이너리 데이터를 NumPy 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 숫자 인식 처리
    cells, rows, cols = extract_digit_cells(image)
    batch_tensor = process_cells_to_tensors(cells)
    predictions = predict_digits(batch_tensor)
    result_grid = predictions.reshape((rows, cols))
    
    return {
        'success': True,
        'rows': rows,
        'cols': cols,
        'grid': result_grid.tolist(),
    }

@app.route('/recognize', methods=['POST'])
def recognize_digit():
    """이미지를 받아 바로 숫자 인식 결과 반환"""
    # 요청에 파일이 있는지 확인
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400

    file = request.files['image']
    file_bytes = file.read()
    
    try:
        # 이미지 처리 및 숫자 인식
        result = process_image(file_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/find-strategy', methods=['POST'])
def find_strategy_api():
    """
    JSON으로 2차원 배열(grid)을 받아 전략을 찾아 반환
    
    요청 형식:
    {
        "grid": [[2, 7, 7, ...], [7, 8, 3, ...], ...]
    }
    """
    # 요청에 JSON 데이터가 있는지 확인
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    # grid 필드가 있는지 확인
    if 'grid' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing grid data'
        }), 400
    
    try:
        # JSON에서 grid 가져오기
        grid = np.array(data['grid'], dtype=np.uint32)
        
        # grid의 형태 유효성 검사
        if len(grid.shape) != 2:
            return jsonify({
                'success': False,
                'error': 'Grid must be a 2D array'
            }), 400
        
        # find_strategy 함수 호출
        strategy = find_strategy(grid)
        print(f"찾은 최고 전략의 점수: {strategy.score}")
        print(f"이동 횟수: {len(strategy.boxes)}")
        
        # 각 이동 출력
        for i, box in enumerate(strategy.boxes):
            print(f"이동 {i+1}: ({box.x}, {box.y}) 위치에서 {box.width}x{box.height} 크기의 영역 제거")
        return jsonify({
            'success': True,
            'strategy': strategy.to_dict()
        })
    
    except ValueError as e:
        # 배열 변환 문제
        return jsonify({
            'success': False,
            'error': f'Invalid grid format: {str(e)}'
        }), 400
    except Exception as e:
        # 기타 모든 예외
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)