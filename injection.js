
(function () {
  /**
   * 인식 결과 객체 타입 정의
   * @typedef {Object} RecognizeResult
   * @property {boolean} success - 인식 성공 여부
   * @property {number} rows - 그리드 행 수
   * @property {number} cols - 그리드 열 수
   * @property {number[][]} grid - 인식된 숫자 그리드
   */

  /**
   * 박스 객체를 나타내는 타입
   * @typedef {Object} Box
   * @property {number} x - 박스의 x 좌표 (가로 위치)
   * @property {number} y - 박스의 y 좌표 (세로 위치)
   * @property {number} width - 박스의 너비
   * @property {number} height - 박스의 높이
   */

  /**
   * 전략 객체를 나타내는 타입
   * @typedef {Object} Strategy
   * @property {Box[]} boxes - 전략에 포함된 박스들의 배열
   * @property {number} score - 전략의 점수
   */

  /**
   * 서버 응답 객체를 나타내는 타입
   * @typedef {Object} StrategyResponse
   * @property {boolean} success - 요청 성공 여부
   * @property {Strategy} strategy - 전략 정보
   */

  /**
   * 드래그 옵션 객체 타입
   * @typedef {Object} DragOptions
   * @property {number} [cropTop=74] - 이미지 상단 크롭 위치
   * @property {number} [cropBottom=400] - 이미지 하단 크롭 위치
   * @property {number} [cropLeft=72] - 이미지 왼쪽 크롭 위치
   * @property {number} [cropRight=628] - 이미지 오른쪽 크롭 위치
   * @property {number} [cellSize=33] - 셀 크기
   * @property {number} [steps=10] - 드래그 단계 수
   * @property {number} [moveDelayMs=50] - 각 단계 간 지연 시간 (밀리초)
   */

  /**
   * 드래그 위치 정보 객체 타입
   * @typedef {Object} DragInfo
   * @property {{x: number, y: number}} gridStart - 그리드 시작 좌표
   * @property {{x: number, y: number}} gridEnd - 그리드 종료 좌표
   * @property {{x: number, y: number}} pixelStart - 픽셀 시작 좌표
   * @property {{x: number, y: number}} pixelEnd - 픽셀 종료 좌표
   */

  /**
   * 기존 코드를 함수로 분리
   * 봇의 주요 로직을 실행하는 비동기 함수
   * @returns {Promise<void>} 실행 완료 Promise
   */

  /** @type {string} 서버 URL */
  const serverUrl = 'http://localhost:5000';

  /**
   * 캔버스 요소를 찾아 반환
   * @returns {HTMLCanvasElement} 찾은 캔버스 요소
   * @throws {Error} 캔버스를 찾을 수 없을 경우 에러
   */
  function getCanvasElement() {
    // 페이지의 모든 캔버스 찾기
    const canvases = document.querySelectorAll('canvas');
    if (canvases.length === 0) {
      throw new Error('캔버스를 찾을 수 없습니다.');
    }
    const canvas = canvases[0];
    return canvas;
  }

  /**
   * 캔버스 요소에서 이미지 Blob을 비동기적으로 추출
   * @param {HTMLCanvasElement} canvasElement - 이미지를 추출할 캔버스 요소
   * @returns {Promise<Blob>} - 캔버스 이미지의 Blob 객체
   * @throws {Error} Blob 생성 실패 시 에러
   */
  async function getCanvasImage(canvasElement) {
    return new Promise((resolve, reject) => {
      try {
        canvasElement.toBlob((blob) => {
          if (!blob) {
            reject(new Error('캔버스에서 Blob을 생성할 수 없습니다.'));
            return;
          }
          resolve(blob);
        }, 'image/png');
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * 캔버스 이미지 Blob을 서버로 업로드하여 인식
   * @param {Blob} canvasImage - 전송할 캔버스 이미지 Blob
   * @returns {Promise<RecognizeResult>} - 인식 결과 객체
   * @throws {Error} 서버 오류 발생 시 예외
   */
  async function recognizeImage(canvasImage) {
    const formData = new FormData();
    formData.append('image', canvasImage, 'canvas-image.png');
    const response = await fetch(`${serverUrl}/recognize`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * 전략을 찾는 함수
   * @param {number[][]} grid - 2차원 숫자 배열 (그리드)
   * @returns {Promise<StrategyResponse>} 전략 정보를 포함한 응답
   * @throws {Error} 서버 오류 발생 시 예외
   */
  async function findStrategy(grid) {
    const response = await fetch(`${serverUrl}/find-strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ grid })
    });

    if (!response.ok) {
      throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * 지정된 시간만큼 대기하는 함수
   * @param {number} ms - 대기 시간 (밀리초)
   * @returns {Promise<void>} 대기 완료 Promise
   */
  async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * 캔버스에서 특정 영역을 드래그하는 비동기 함수
   * @param {number} x1 - 드래그 시작점 X 좌표
   * @param {number} y1 - 드래그 시작점 Y 좌표
   * @param {number} x2 - 드래그 끝점 X 좌표
   * @param {number} y2 - 드래그 끝점 Y 좌표
   * @param {DragOptions} [options={}] - 추가 옵션
   * @returns {Promise<DragInfo>} 드래그 완료 후 결과 객체를 포함한 Promise
   */
  async function dragCanvas(x1, y1, x2, y2, options = {}) {
    // 기본 옵션값 설정 (파이썬 코드의 값들을 기본값으로 사용)
    const {
      cropTop = 74,
      cropBottom = 400,
      cropLeft = 72,
      cropRight = 628,
      cellSize = 33,
      steps = 10,
      moveDelayMs = 50
    } = options;

    // 캔버스의 위치 정보 가져오기
    const canvas = getCanvasElement();
    const rect = canvas.getBoundingClientRect();

    /**
     * 좌표 변환: 그리드 좌표를 실제 픽셀 좌표로 변환
     * @param {number} gridX - 그리드 X 좌표
     * @param {number} gridY - 그리드 Y 좌표
     * @returns {{x: number, y: number}} 픽셀 좌표
     */
    const gridToPixel = (gridX, gridY) => {
      return {
        x: cropLeft + (gridX * cellSize),
        y: cropTop + (gridY * cellSize)
      };
    };

    // 그리드 좌표를 픽셀 좌표로 변환
    const startPos = gridToPixel(x1, y1);
    const endPos = gridToPixel(x2, y2);

    // 브라우저 창 기준 절대 좌표로 변환
    const absStartX = rect.left + startPos.x;
    const absStartY = rect.top + startPos.y;
    const absEndX = rect.left + endPos.x;
    const absEndY = rect.top + endPos.y;

    // 드래그 정보 객체
    /** @type {DragInfo} */
    const dragInfo = {
      gridStart: { x: x1, y: y1 },
      gridEnd: { x: x2, y: y2 },
      pixelStart: startPos,
      pixelEnd: endPos
    };

    // 마우스 다운 이벤트 발생
    const mouseDownEvent = createMouseEvent('mousedown', absStartX, absStartY);
    canvas.dispatchEvent(mouseDownEvent);

    // 마우스 무브 이벤트 시뮬레이션
    for (let i = 1; i <= steps; i++) {
      await sleep(moveDelayMs);

      const ratio = i / steps;
      const currentX = absStartX + (absEndX - absStartX) * ratio;
      const currentY = absStartY + (absEndY - absStartY) * ratio;

      const mouseMoveEvent = createMouseEvent('mousemove', currentX, currentY);
      canvas.dispatchEvent(mouseMoveEvent);
    }

    // 마지막에 mouseup 이벤트 발생
    await sleep(moveDelayMs);
    const mouseUpEvent = createMouseEvent('mouseup', absEndX, absEndY);
    canvas.dispatchEvent(mouseUpEvent);

    // 드래그 정보 반환
    return dragInfo;
  }

  /**
   * 마우스 이벤트를 생성하는 유틸리티 함수
   * @param {string} type - 이벤트 타입 (mousedown, mousemove, mouseup)
   * @param {number} x - X 좌표
   * @param {number} y - Y 좌표
   * @returns {MouseEvent} 생성된 마우스 이벤트
   */
  function createMouseEvent(type, x, y) {
    return new MouseEvent(type, {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: x,
      clientY: y
    });
  }

  /**
   * 메인 로직 실행 함수
   * @returns {Promise<void>}
   */
  async function botLogic() {
    try {
      updateStatus('캔버스 이미지 인식 중...');
      // 캔버스 이미지 업로드 (인식)
      const canvas = getCanvasElement();
      const blob = await getCanvasImage(canvas);
      const recognized = await recognizeImage(blob);

      updateStatus('전략 찾는 중...');
      // 전략 찾기
      const result = await findStrategy(recognized.grid);

      updateStatus(`전략 실행 중... (총 ${result.strategy.boxes.length}개 박스)`);
      let completedBoxes = 0;

      for (let box of result.strategy.boxes) {
        const { x, y, width, height } = box;
        await dragCanvas(x, y, x + width, y + height, {
          steps: 1,
          moveDelayMs: 10
        });
        completedBoxes++;
        updateStatus(`전략 실행 중... (${completedBoxes}/${result.strategy.boxes.length})`);
      }

      updateStatus('완료! 점수: ' + result.strategy.score);
    } catch (error) {
      console.error(error);
      updateStatus('오류 발생: ' + error.message, true);
    }
  }


  /**
   * UI 상태 업데이트 함수
   * @param {string} message - 표시할 메시지
   * @param {boolean} [isError=false] - 오류 메시지 여부
   */
  function updateStatus(message, isError = false) {
    const statusEl = document.getElementById('bot-status');
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.style.color = isError ? '#f44336' : '#4CAF50';
    }
  }

  // UI가 이미 존재하는지 확인
  if (document.getElementById('bot-control-panel')) {
    console.log('봇 컨트롤 패널이 이미 존재합니다.');
    return;
  }

  /**
   * 컨트롤 패널 요소 생성
   * @type {HTMLDivElement}
   */
  const controlPanel = document.createElement('div');
  controlPanel.id = 'bot-control-panel';
  controlPanel.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    padding: 15px;
    z-index: 9999;
    width: 220px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    transition: all 0.3s ease;
    border: 1px solid #e1e4e8;
  `;

  /**
   * 제목 요소 생성
   * @type {HTMLDivElement}
   */
  const title = document.createElement('div');
  title.textContent = '🤖 봇 컨트롤러';
  title.style.cssText = `
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 12px;
    color: #333;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  `;

  /**
   * 최소화 버튼 생성
   * @type {HTMLSpanElement}
   */
  const minimizeBtn = document.createElement('span');
  minimizeBtn.innerHTML = '−';
  minimizeBtn.style.cssText = `
    cursor: pointer;
    font-size: 18px;
    color: #666;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
  `;
  minimizeBtn.title = '최소화';
  minimizeBtn.addEventListener('mouseenter', () => {
    minimizeBtn.style.backgroundColor = '#f5f5f5';
  });
  minimizeBtn.addEventListener('mouseleave', () => {
    minimizeBtn.style.backgroundColor = 'transparent';
  });

  /** @type {boolean} 최소화 상태 */
  let minimized = false;
  minimizeBtn.addEventListener('click', () => {
    const content = document.getElementById('bot-control-content');
    if (minimized) {
      content.style.display = 'block';
      minimizeBtn.innerHTML = '−';
      controlPanel.style.width = '220px';
    } else {
      content.style.display = 'none';
      minimizeBtn.innerHTML = '+';
      controlPanel.style.width = '150px';
    }
    minimized = !minimized;
  });

  title.appendChild(minimizeBtn);
  controlPanel.appendChild(title);

  /**
   * 컨텐츠 컨테이너 생성
   * @type {HTMLDivElement}
   */
  const content = document.createElement('div');
  content.id = 'bot-control-content';

  /**
   * 상태 표시 요소 생성
   * @type {HTMLDivElement}
   */
  const status = document.createElement('div');
  status.id = 'bot-status';
  status.textContent = '준비 완료';
  status.style.cssText = `
    margin-bottom: 12px;
    font-size: 14px;
    color: #4CAF50;
    background-color: rgba(76, 175, 80, 0.1);
    padding: 8px;
    border-radius: 4px;
    word-break: break-word;
  `;

  /**
   * 실행 버튼 생성
   * @type {HTMLButtonElement}
   */
  const runButton = document.createElement('button');
  runButton.textContent = '봇 실행';
  runButton.style.cssText = `
    width: 100%;
    padding: 10px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
  `;

  runButton.addEventListener('mouseenter', () => {
    runButton.style.backgroundColor = '#45a049';
    runButton.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
  });

  runButton.addEventListener('mouseleave', () => {
    runButton.style.backgroundColor = '#4CAF50';
    runButton.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.1)';
  });

  runButton.addEventListener('click', async () => {
    runButton.disabled = true;
    runButton.style.backgroundColor = '#cccccc';
    runButton.textContent = '실행 중...';

    try {
      await botLogic();
    } catch (error) {
      console.error('봇 실행 오류:', error);
      updateStatus('오류 발생: ' + error.message, true);
    } finally {
      runButton.disabled = false;
      runButton.style.backgroundColor = '#4CAF50';
      runButton.textContent = '봇 실행';
    }
  });

  // 드래그 가능하게 만들기
  /** @type {boolean} 드래그 중 여부 */
  let isDragging = false;
  /** @type {number} 드래그 시작 X 오프셋 */
  let offsetX;
  /** @type {number} 드래그 시작 Y 오프셋 */
  let offsetY;

  title.addEventListener('mousedown', (e) => {
    if (e.target === minimizeBtn) return;
    isDragging = true;
    offsetX = e.clientX - controlPanel.getBoundingClientRect().left;
    offsetY = e.clientY - controlPanel.getBoundingClientRect().top;
    title.style.cursor = 'grabbing';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const x = e.clientX - offsetX;
    const y = e.clientY - offsetY;

    // 화면 내에 유지
    const maxX = window.innerWidth - controlPanel.offsetWidth;
    const maxY = window.innerHeight - controlPanel.offsetHeight;

    controlPanel.style.left = Math.max(0, Math.min(x, maxX)) + 'px';
    controlPanel.style.top = Math.max(0, Math.min(y, maxY)) + 'px';
    controlPanel.style.right = 'auto';
  });

  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      title.style.cursor = 'grab';
    }
  });

  title.style.cursor = 'grab';

  // UI 조립 및 문서에 추가
  content.appendChild(status);
  content.appendChild(runButton);
  controlPanel.appendChild(content);
  document.body.appendChild(controlPanel);

  console.log('봇 컨트롤 패널이 추가되었습니다.');
})();