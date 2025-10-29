from __future__ import annotations
import json
from typing import Optional
import cv2
import numpy as np
from collections import defaultdict
import os
import time
from api_handler import ApiHandler
from redis_manager import RedisPublisher, RedisCache
import base64
import grpc
from grpc_dir import led_pb2, led_pb2_grpc
from env_manager import get_env
import signal
import sys

def shutdown(sig, frame):
    print("SIGINT received, exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

RPC_PORT = int(get_env("RPC_PORT", "-1"))
RPC_HOST = get_env("RPC_HOST", "localhost")
RPC_ADDRESS = f"{RPC_HOST}:{RPC_PORT}"
THRESHOLD = float(get_env("THRESHOLD", 50.0))

SOURCE_URL = get_env("SOURCE_URL", None)
SOURCE_PORT = int(get_env("SOURCE_PORT", "-1"))
RTSP_ID_PWD = get_env("RTSP_ID_PWD", None)
RTSP_CONNECTION_URL = get_env("RTSP_CONNECTION_URL", None)
RTSP_URL = f"rtsp://{RTSP_ID_PWD}@{RTSP_CONNECTION_URL}:{SOURCE_PORT}"
DESIRED_FPS = int(os.getenv("DESIRED_FPS", 1))
DRF_BASE_URL = os.getenv("DRF_BASE_URL")
DRF_LOGIN_URL = os.getenv("DRF_LOGIN_URL")
DRF_LOGIN_INFO = json.loads(os.getenv("DRF_LOGIN_INFO" ))

DRF_DATA_URL = get_env("DRF_DATA_URL")


### flask health check
from flask import Flask, jsonify

app = Flask(__name__)
@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify({"health_check": "ok"})


def check_env():
    if not SOURCE_URL:
        raise ValueError("SOURCE_URL is not set")
    if not SOURCE_PORT:
        raise ValueError("SOURCE_PORT is not set")
    if not RTSP_ID_PWD:
        raise ValueError("RTSP_ID_PWD is not set")
    if not RTSP_CONNECTION_URL:
        raise ValueError("RTSP_CONNECTION_URL is not set")
    if not RTSP_URL:
        raise ValueError("RTSP_URL is not set")
    if not DESIRED_FPS:
        raise ValueError("DESIRED_FPS is not set")
    if not DRF_BASE_URL:
        raise ValueError("DRF_BASE_URL is not set")
    if not DRF_LOGIN_URL:
        raise ValueError("DRF_LOGIN_URL is not set")
    if not DRF_LOGIN_INFO:
        raise ValueError("DRF_LOGIN_INFO is not set")
    if not DRF_DATA_URL:
        raise ValueError("DRF_DATA_URL is not set")
    if not RPC_ADDRESS:
        raise ValueError("RPC_ADDRESS is not set")
    if not THRESHOLD:
        raise ValueError("THRESHOLD is not set")
    if not RPC_PORT:
        raise ValueError("RPC_PORT is not set")
    if not RPC_HOST:
        raise ValueError("RPC_HOST is not set")
    if not RPC_ADDRESS:
        raise ValueError("RPC_ADDRESS is not set")
    if not THRESHOLD:
        raise ValueError("THRESHOLD is not set")
    if not RPC_PORT:
        raise ValueError("RPC_PORT is not set")
    if not RPC_HOST:
        raise ValueError("RPC_HOST is not set")
    
    print("================================================")
    print(f"Environment Variables:")
    print(f"SOURCE_URL: {SOURCE_URL}")
    print(f"SOURCE_PORT: {SOURCE_PORT}")
    print(f"RTSP_ID_PWD: {RTSP_ID_PWD}")
    print(f"RTSP_CONNECTION_URL: {RTSP_CONNECTION_URL}")
    print(f"RTSP_URL: {RTSP_URL}")
    print(f"DESIRED_FPS: {DESIRED_FPS}")
    print(f"DRF_BASE_URL: {DRF_BASE_URL}")
    print(f"DRF_LOGIN_URL: {DRF_LOGIN_URL}")
    print(f"DRF_LOGIN_INFO: {DRF_LOGIN_INFO}")
    print(f"DRF_DATA_URL: {DRF_DATA_URL}")
    print(f"RPC_ADDRESS: {RPC_ADDRESS}")
    print(f"THRESHOLD: {THRESHOLD}")
    print(f"RPC_PORT: {RPC_PORT}")
    print(f"RPC_HOST: {RPC_HOST}")
    print("================================================")


def get_drf_data(rtsp_url:str, api_handler:ApiHandler) -> Optional[dict]:
	# redis_cache = RedisCache(redis_host="localhost", redis_port="6379", redis_db=1)
    from redis_manager import RedisCache
    redis_cache = RedisCache()
    cache_key = f"DRF:RTSPCameraSetting:{rtsp_url}"
    cache_data = redis_cache.get(cache_key)
    if cache_data:
        drf_data:dict = json.loads(cache_data)
        if drf_data and isinstance(drf_data, dict) and drf_data.get("url") == rtsp_url:
            return drf_data
        else:
            print (f" Cache Data is not valid: {drf_data} {drf_data.get('url')} != {rtsp_url}")
            return None
    else:
        #### 전체 data 조회
        response = api_handler.get(DRF_DATA_URL, params={"page_size": 0, "is_active": True, "url": rtsp_url } )
        if response.ok:
            for data in response.json():
                if data["url"] == rtsp_url:
                    drf_data = data
                    break
            redis_cache.set(cache_key, json.dumps(drf_data))
            return drf_data
        else:
            print(f"DRF 데이터 조회 실패: {response.status_code} {response.text}")
            return None



last_analyze_time = 0.0
def valid_fps_time():
    global last_analyze_time
    now = time.time()
    if now - last_analyze_time < 1.0 / DESIRED_FPS:
        return False
    last_analyze_time = now
    return True

def run_grpc(
    source:str="image", 
    image:np.ndarray=None, 
    path:str=None, 
    rois:dict={}, 
    map_led:dict=None, 
    threshold:float=None) -> dict:
    threshold = threshold if threshold is not None else THRESHOLD
    channel = grpc.insecure_channel(
        RPC_ADDRESS,
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    stub = led_pb2_grpc.LEDAnalyzerServiceStub(channel)

    try:
        # 이미지 준비
        if source == "image":
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Image invalid")
            ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                raise ValueError("Encode failed")
            image_bytes = buffer.tobytes()
        elif source == "path":
            if not path or not os.path.exists(path):
                raise ValueError("Path invalid")
            with open(path, "rb") as f:
                image_bytes = f.read()
        else:
            raise ValueError(f"Invalid source: {source}")

        # 요청 전송
        request = led_pb2.RunRequest(
            source="image",
            image_bytes=image_bytes,
            all_roi_dict=json.dumps(rois),
            map_led=json.dumps(map_led),
            threshold=threshold
        )

        response = stub.Run(request, timeout=5)
        results = json.loads(response.final_results)
        ### 10-29변경 : 기존 grpc에서 image까지 받는것을 분석만 받고, 이미지는 의뢰한데서 처리함.

        return {"ok": True, "results": results , 'image': image_post_process(image)}#, "image": image_b64}

    except (grpc.RpcError, ValueError) as e:
        # 연결 실패, 타임아웃, 인코딩 오류 등 모두 여기서 잡힘
        return {"ok": False, "error": str(e)}


def image_post_process(image: np.ndarray) -> str:
    """ 1. fhd 급 resize
        2. jpeg 인코딩
    """
    resized = cv2.resize(image, (1920, 1080))
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


from datetime import datetime
def main():
    check_env()
    r = RedisPublisher()
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {RTSP_URL}")

    api_handler = ApiHandler(base_url=DRF_BASE_URL)
    if not api_handler.login(DRF_LOGIN_URL, DRF_LOGIN_INFO):
        print(f"DRF 로그인 실패: {DRF_LOGIN_URL} {DRF_LOGIN_INFO}")
        exit(1)

    drf_setting_data = get_drf_data(SOURCE_URL, api_handler)
    if not drf_setting_data:
        print(f"DRF 데이터 조회 실패: {drf_setting_data}")
        exit(1)
    else:
        print (f"DRF 데이터 조회 성공: {drf_setting_data}")

    while True and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        if valid_fps_time():
            # print ( f"{datetime.now()} frame: {frame.shape} {frame.size}")
            # result, image = analyzer.run(frame, drf_setting_data.get("rois", {}), drf_setting_data.get("map_led", {}))
            #         # FHD로 리사이즈
            # resized = cv2.resize(image, (1920, 1080))

            # # JPEG 인코딩 후 bytes 변환
            # ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            # if ok:
            #     image_bytes = buffer.tobytes()
            #     image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            # else:
            #     image_b64 = None
            start_time = time.perf_counter()
            result = run_grpc(source="image", image=frame, rois=drf_setting_data.get("rois", {}), map_led=drf_setting_data.get("map_led", {}))
            if not result.get("ok"):
                print (f"run_grpc error: {json.dumps(result, indent=4)}")
                continue
            # 메시지 구성 (result + 이미지)
            message = { SOURCE_URL: result }
            r.publish(message=message)
            print (f"run_grpc time:  rtsp_url: {SOURCE_URL}:{SOURCE_PORT} {1000*(time.perf_counter() - start_time)} msec")
            # print (f"result: {result}")




if __name__ == '__main__':
    import threading
    threading.Thread(target=app.run, args=("0.0.0.0", 5000), daemon=True).start()
    main()