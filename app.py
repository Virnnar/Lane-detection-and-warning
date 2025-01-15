from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def detect_lanes(frame):
    height, width, _ = frame.shape

    region_of_interest_vertices = [
        (0, height),
        (width // 2, int(height * 0.6)),
        (width, height)
    ]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    edges = cv2.Canny(blurred_frame, 50, 150)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)

    # 绘制车道线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(frame, "Lanes detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No lanes detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# 实时摄像头流处理
@app.route('/live')
def live_video():
    def generate_frames():
        cap = cv2.VideoCapture(0)  # 打开摄像头
        if not cap.isOpened():
            raise RuntimeError("无法访问摄像头")

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 调用车道检测函数
            processed_frame = detect_lanes(frame)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # 保存
    file_path = os.path.join(UPLOAD_FOLDER, 'test.mp4')
    file.save(file_path)

    try:
        result = subprocess.run(['main.exe'], cwd=UPLOAD_FOLDER, capture_output=True, text=True)
        return jsonify({'message': 'Execution complete', 'output': result.stdout})
    except Exception as e:
        return jsonify({'message': f'Error running main.exe: {str(e)}'}), 500

# 前端主页
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)








