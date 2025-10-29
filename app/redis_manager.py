from __future__ import annotations

from typing import Dict, Any        
import redis
import json
from datetime import datetime
import numpy as np
import os
from env_manager import get_env

REDIS_HOST = get_env("REDIS_HOST")
REDIS_PORT = int(get_env("REDIS_PORT"))
REDIS_DB_FOR_CACHE = int(get_env("REDIS_DB_FOR_CACHE"))
REDIS_DB_FOR_CHANNEL = int(get_env("REDIS_DB_FOR_CHANNEL"))
REDIS_PASSWORD = get_env("REDIS_PASSWORD")
REDIS_CHANNEL = get_env("REDIS_CHANNEL")


def check_env():
    if not REDIS_HOST:
        raise ValueError("REDIS_HOST is not set")
    if not REDIS_PORT:
        raise ValueError("REDIS_PORT is not set")
    if not (isinstance(REDIS_DB_FOR_CACHE, int) and REDIS_DB_FOR_CACHE >= 0):
        raise ValueError("REDIS_DB_FOR_CACHE is not set")
    if not (isinstance(REDIS_DB_FOR_CHANNEL, int) and REDIS_DB_FOR_CHANNEL >= 0):
        raise ValueError("REDIS_DB_FOR_CHANNE is not set")
    if not REDIS_PASSWORD:
        raise ValueError("REDIS_PASSWORD is not set")
    if not REDIS_CHANNEL:
        raise ValueError("REDIS_CHANNEL is not set")


def to_serializable(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


check_env()

class RedisCache:
    def __init__(self, redis_host:str=REDIS_HOST, redis_port:str=REDIS_PORT, redis_db:int=REDIS_DB_FOR_CACHE, redis_password:str=REDIS_PASSWORD):
        self.redis_host = redis_host
        self.redis_port = int(redis_port)
        self.redis_db = int(redis_db)
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=redis_password
        )
    
    def get(self, key: str) -> str:
        return self.redis_client.get(key)
    
    def set(self, key: str, value: str) -> None:
        self.redis_client.set(key, value)

class RedisPublisher:
    """
    Redis를 통해 메시지를 발행하는 클래스
    """
    
    def __init__(self, host:str=REDIS_HOST, port:str=REDIS_PORT, db:int=REDIS_DB_FOR_CHANNEL, channel:str=REDIS_CHANNEL, password:str=REDIS_PASSWORD):
        """
        Redis 연결 초기화
        
        Args:
            redis_host: Redis 호스트 (기본값:"localhost"
            redis_port: Redis 포트 (기본값: "6379")
            redis_db: Redis DB 번호 (기본값: 0)
        """
        self.redis_host = host
        self.redis_port = int(port)
        self.redis_db = int(db)
        self.redis_channel = channel 

        # Redis 클라이언트 초기화
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=password
        )
    
    def publish(self, channel: str=None, message: Dict[str, Any] = {}) -> int:
        """
        지정된 채널에 메시지 발행
        
        Args:
            channel: 메시지를 발행할 채널 이름
            message: 발행할 메시지 데이터 (딕셔너리)
            
        Returns:
            구독자 수
        """
        try:
            # 메시지를 JSON 문자열로 직렬화
            msg = self.get_default_msg()
            msg['message'] = message
            message_str = json.dumps(msg, default=to_serializable, ensure_ascii=False)

            channel = channel or self.redis_channel
            # print(f"메시지 발행: {channel} {message_str}")
            # 메시지 발행
            subscribers = self.redis_client.publish(channel, message_str)
            return subscribers
        except Exception as e:
            print(f"메시지 발행 중 오류 발생: {e}")
            return 0

    def get_default_msg(self) -> Dict[str, Any]:
        return {
            'main_type': 'update',
            'sub_type': 'digit_analysis',
            'action' :'update',
            'message' : {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'receiver': 'all',
            'sender': 'hi_rtsp_digit_anaylizer'
        }