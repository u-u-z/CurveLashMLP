let isDrawing = false;
let points = [];
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');

// Setup canvas
ctx.strokeStyle = 'blue';
ctx.lineWidth = 2;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDrawing);
canvas.addEventListener('mouseout', endDrawing);
canvas.addEventListener('touchstart', handleTouch);
canvas.addEventListener('touchmove', handleTouch);
canvas.addEventListener('touchend', endDrawing);

document.getElementById('clearButton').addEventListener('click', clearCanvas);

// SVG 控制点模式代码
const svgCanvas = document.getElementById('svgCanvas');
const curvePath = document.getElementById('curvePath');
const predictedPath = document.getElementById('predictedPath');
let controlPoints = [];
let selectedPoint = null;
let isDragging = false;

// 添加事件监听
document.getElementById('addPointButton').addEventListener('click', addControlPoint);
document.getElementById('clearPointsButton').addEventListener('click', clearControlPoints);

// SVG 命名空间
const SVG_NS = "http://www.w3.org/2000/svg";

// 添加控制点
function addControlPoint() {
    const x = 100 + controlPoints.length * 50;
    const y = 200;
    
    const group = document.createElementNS(SVG_NS, "g");
    
    // 主控制点
    const point = document.createElementNS(SVG_NS, "circle");
    point.setAttribute("cx", x);
    point.setAttribute("cy", y);
    point.setAttribute("r", 6);
    point.setAttribute("class", "control-point");
    point.setAttribute("data-type", "main");
    
    // 前控制手柄
    const handle1 = document.createElementNS(SVG_NS, "circle");
    handle1.setAttribute("cx", x - 50);
    handle1.setAttribute("cy", y);
    handle1.setAttribute("r", 4);
    handle1.setAttribute("class", "handle-point");
    handle1.setAttribute("data-type", "handle1");
    
    // 后控制手柄
    const handle2 = document.createElementNS(SVG_NS, "circle");
    handle2.setAttribute("cx", x + 50);
    handle2.setAttribute("cy", y);
    handle2.setAttribute("r", 4);
    handle2.setAttribute("class", "handle-point");
    handle2.setAttribute("data-type", "handle2");
    
    // 控制线
    const line1 = document.createElementNS(SVG_NS, "line");
    line1.setAttribute("x1", x - 50);
    line1.setAttribute("y1", y);
    line1.setAttribute("x2", x);
    line1.setAttribute("y2", y);
    line1.setAttribute("class", "handle-line");
    
    const line2 = document.createElementNS(SVG_NS, "line");
    line2.setAttribute("x1", x);
    line2.setAttribute("y1", y);
    line2.setAttribute("x2", x + 50);
    line2.setAttribute("y2", y);
    line2.setAttribute("class", "handle-line");
    
    group.appendChild(line1);
    group.appendChild(line2);
    group.appendChild(handle1);
    group.appendChild(handle2);
    group.appendChild(point);
    
    // 添加拖拽事件
    [point, handle1, handle2].forEach(elem => {
        elem.addEventListener('mousedown', startDragging);
        elem.addEventListener('touchstart', handlePointTouch);
    });
    
    svgCanvas.appendChild(group);
    
    controlPoints.push({
        group: group,
        point: point,
        handle1: handle1,
        handle2: handle2,
        line1: line1,
        line2: line2,
        x: x,
        y: y,
        h1x: x - 50,
        h1y: y,
        h2x: x + 50,
        h2y: y
    });
    
    updateCurve();
}

// 清除所有控制点
function clearControlPoints() {
    controlPoints.forEach(cp => {
        cp.group.remove();
    });
    controlPoints = [];
    curvePath.setAttribute('d', '');
    predictedPath.setAttribute('d', '');
}

// 开始拖动
function startDragging(e) {
    selectedPoint = e.target;
    isDragging = true;
    
    document.addEventListener('mousemove', dragPoint);
    document.addEventListener('mouseup', stopDragging);
}

// 处理触摸事件
function handlePointTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    selectedPoint = e.target;
    isDragging = true;
    
    document.addEventListener('touchmove', dragPointTouch);
    document.addEventListener('touchend', stopDragging);
}

// 拖动点
function dragPoint(e) {
    if (!isDragging) return;
    
    const rect = svgCanvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    
    // 限制在画布范围内
    x = Math.max(6, Math.min(x, svgCanvas.width.baseVal.value - 6));
    y = Math.max(6, Math.min(y, svgCanvas.height.baseVal.value - 6));
    
    updatePointPosition(x, y);
}

// 处理触摸拖动
function dragPointTouch(e) {
    if (!isDragging) return;
    
    const touch = e.touches[0];
    const rect = svgCanvas.getBoundingClientRect();
    let x = touch.clientX - rect.left;
    let y = touch.clientY - rect.top;
    
    // 限制在画布范围内
    x = Math.max(6, Math.min(x, svgCanvas.width.baseVal.value - 6));
    y = Math.max(6, Math.min(y, svgCanvas.height.baseVal.value - 6));
    
    updatePointPosition(x, y);
}

// 更新点位置
function updatePointPosition(x, y) {
    const type = selectedPoint.getAttribute('data-type');
    const pointData = controlPoints.find(cp => 
        cp.point === selectedPoint || 
        cp.handle1 === selectedPoint || 
        cp.handle2 === selectedPoint
    );
    
    if (!pointData) return;
    
    if (type === 'main') {
        // 移动主控制点时，同时移动手柄
        const dx = x - pointData.x;
        const dy = y - pointData.y;
        
        pointData.x = x;
        pointData.y = y;
        pointData.h1x += dx;
        pointData.h1y += dy;
        pointData.h2x += dx;
        pointData.h2y += dy;
        
        pointData.point.setAttribute('cx', x);
        pointData.point.setAttribute('cy', y);
        pointData.handle1.setAttribute('cx', pointData.h1x);
        pointData.handle1.setAttribute('cy', pointData.h1y);
        pointData.handle2.setAttribute('cx', pointData.h2x);
        pointData.handle2.setAttribute('cy', pointData.h2y);
        
        // 更新控制线
        pointData.line1.setAttribute('x1', pointData.h1x);
        pointData.line1.setAttribute('y1', pointData.h1y);
        pointData.line1.setAttribute('x2', x);
        pointData.line1.setAttribute('y2', y);
        pointData.line2.setAttribute('x1', x);
        pointData.line2.setAttribute('y1', y);
        pointData.line2.setAttribute('x2', pointData.h2x);
        pointData.line2.setAttribute('y2', pointData.h2y);
    } else if (type === 'handle1') {
        // 移动前手柄
        pointData.h1x = x;
        pointData.h1y = y;
        pointData.handle1.setAttribute('cx', x);
        pointData.handle1.setAttribute('cy', y);
        pointData.line1.setAttribute('x1', x);
        pointData.line1.setAttribute('y1', y);
    } else if (type === 'handle2') {
        // 移动后手柄
        pointData.h2x = x;
        pointData.h2y = y;
        pointData.handle2.setAttribute('cx', x);
        pointData.handle2.setAttribute('cy', y);
        pointData.line2.setAttribute('x2', x);
        pointData.line2.setAttribute('y2', y);
    }
    
    updateCurve();
}

// 停止拖动
function stopDragging() {
    if (!isDragging) return;
    
    isDragging = false;
    selectedPoint = null;
    
    document.removeEventListener('mousemove', dragPoint);
    document.removeEventListener('mouseup', stopDragging);
    document.removeEventListener('touchmove', dragPointTouch);
    document.removeEventListener('touchend', stopDragging);
    
    // 预测曲线B
    if (controlPoints.length >= 2) {
        predictFromControlPoints();
    }
}

// 更新曲线
function updateCurve() {
    if (controlPoints.length < 2) {
        curvePath.setAttribute('d', '');
        return;
    }
    
    let d = `M ${controlPoints[0].x} ${controlPoints[0].y}`;
    
    for (let i = 0; i < controlPoints.length - 1; i++) {
        const curr = controlPoints[i];
        const next = controlPoints[i + 1];
        d += ` C ${curr.h2x},${curr.h2y} ${next.h1x},${next.h1y} ${next.x},${next.y}`;
    }
    
    curvePath.setAttribute('d', d);
}

// 从控制点预测
async function predictFromControlPoints() {
    if (controlPoints.length < 2) return;
    
    // 从贝塞尔曲线上采样点
    const allSamples = [];
    
    // 对每一段曲线进行采样
    for (let i = 0; i < controlPoints.length - 1; i++) {
        const segmentPoints = [
            controlPoints[i],
            { x: controlPoints[i].h2x, y: controlPoints[i].h2y },
            { x: controlPoints[i + 1].h1x, y: controlPoints[i + 1].h1y },
            controlPoints[i + 1]
        ];
        const samples = sampleBezierSegment(segmentPoints, 20);
        // 只添加第一个点（如果是第一段）或者其他所有点
        if (i === 0) {
            allSamples.push(...samples);
        } else {
            allSamples.push(...samples.slice(1));
        }
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                points: allSamples
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // 使用贝塞尔曲线绘制预测结果
        let d = `M ${data.predicted_points[0][0]} ${data.predicted_points[0][1]}`;
        
        // 使用直线连接预测点，以保持连续性
        for (let i = 1; i < data.predicted_points.length; i++) {
            const point = data.predicted_points[i];
            d += ` L ${point[0]},${point[1]}`;
        }
        
        predictedPath.setAttribute('d', d);
    } catch (error) {
        console.error('Error:', error);
        alert('预测失败，请重试');
    }
}

// 采样单个贝塞尔曲线段
function sampleBezierSegment(points, numSamples) {
    const samples = [];
    for (let t = 0; t <= 1; t += 1 / (numSamples - 1)) {
        const point = evaluateCubicBezier(points[0], points[1], points[2], points[3], t);
        samples.push([point.x, point.y]);
    }
    return samples;
}

// 计算三次贝塞尔曲线上的点
function evaluateCubicBezier(p0, p1, p2, p3, t) {
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    const t2 = t * t;
    const t3 = t2 * t;
    
    return {
        x: mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x,
        y: mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y
    };
}

// 自由绘制模式函数
function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function startDrawing(e) {
    isDrawing = true;
    points = [];
    const point = getCanvasPoint(e);
    points.push(point);
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
}

function draw(e) {
    if (!isDrawing) return;
    const point = getCanvasPoint(e);
    points.push(point);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
}

async function endDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    
    if (points.length >= 2) {
        await predictCurve();
    }
}

function getCanvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points = [];
    ctx.strokeStyle = 'blue';
    ctx.beginPath();
}

async function predictCurve() {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                points: points.map(p => [p.x, p.y])
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Draw predicted curve B
        ctx.strokeStyle = 'red';
        ctx.beginPath();
        const predictedPoints = data.predicted_points;
        ctx.moveTo(predictedPoints[0][0], predictedPoints[0][1]);
        
        for (let i = 1; i < predictedPoints.length; i++) {
            ctx.lineTo(predictedPoints[i][0], predictedPoints[i][1]);
        }
        ctx.stroke();
        
        // Reset stroke style for next drawing
        ctx.strokeStyle = 'blue';
    } catch (error) {
        console.error('Error:', error);
        const errorMessage = error.message === 'Network response was not ok' 
            ? '预测失败，请重试'
            : '发生错误，请重新绘制';
        alert(errorMessage);
    }
} 