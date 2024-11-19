import os
import cv2
import pathlib
import requests
from datetime import datetime

class ChangeDetection:
    result_prev = []
    HOST = 'https://samuel26.pythonanywhere.com/'
    username = 'admin'
    password = '1234'
    token = '641ab83796b2582d4ff26009cbad288ace518e69'
    author = 1
    title = ''
    text = ''

    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    def __init__(self, names):
        self.result_prev = [0 for i in range(len(names))]

        res = requests.post(self.HOST + '/api-token-auth/', {
            'username': self.username,
            'password': self.password,
        })
        res.raise_for_status()
        self.token = res.json()['token']  # 토큰 저장
        print(self.token)


    def log_detection_details(self, labels_to_draw):
        """
        탐지된 객체의 좌표와 크기를 출력합니다.
        """
        print("Detection Details:")
        for label, (x1, y1, x2, y2) in labels_to_draw:
            width = int(x2 - x1)
            height = int(y2 - y1)
            print(f"Object: {label}, Coordinates: ({x1}, {y1}, {x2}, {y2}), Size: {width}x{height}")

    def add(self, names, detected_current, save_dir, image, labels_to_draw, detected_class_names, detected_count):
        self.title = ''
        self.text = ''
        change_flag = False

        for i in range(len(self.result_prev)):
            if self.result_prev[i] == 0 and detected_current[i] > 0:  # 감지 상태 변경 확인
                change_flag = True
                self.title = names[i]

        self.result_prev = detected_current[:]

        if change_flag:
            # 감지된 클래스 이름 및 객체 개수 기록
            self.text = f"Detected Classes: {detected_class_names}\n"
            self.text += f"Detected Count: {detected_count}"  # 실제 객체 개수 사용

            # 변경된 상태에서 send 메서드 호출
            self.send(save_dir, image, labels_to_draw, detected_count, [])


    def send(self, save_dir, image, labels_to_draw, detection_count, detection_classes):
        now = datetime.now()
        today = now.date()

        save_path = pathlib.Path(save_dir) / 'detected' / str(today.year) / str(today.month) / str(today.day)
        save_path.mkdir(parents=True, exist_ok=True)

        file_name = f"{now.strftime('%H-%M-%S')}-{now.microsecond}.jpg"
        full_path = save_path / file_name

        # 이미지에 레이블 추가
        for label, (x1, y1, x2, y2) in labels_to_draw:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save resized image
        dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(full_path), dst)

        headers = {
            'Authorization': f'JWT {self.token}',
            'Accept': 'application/json'
        }

        # API에 보낼 데이터
        data = {
            'title': self.title,
            'text': self.text,  # add()에서 생성된 self.text 사용
            'author': self.author,
            'created_date': now.isoformat(),
            'published_date': now.isoformat()
        }

        files = {'image': open(full_path, 'rb')}
        try:
            res = requests.post(f'{self.HOST}/api_root/Post/', data=data, files=files, headers=headers)
            print(f"Response: {res.status_code}, {res.text}")
        except requests.RequestException as e:
            print(f"Failed to send data: {e}")

        # 특정 임계값 초과 시 경고 출력
        self.check_detection_threshold(detection_count)


# 추가된 메서드: 특정 임계값 초과 시 경고 출력
    def check_detection_threshold(self, detection_count):
        threshold = 10  # 임계값 설정
        if detection_count > threshold:
            print(f"Warning: Detected object count exceeds the threshold! ({detection_count} > {threshold})")
            # 추가 작업을 수행할 수 있음 (예: 알림, 로그 저장 등)