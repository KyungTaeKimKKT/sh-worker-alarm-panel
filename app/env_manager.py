import os
import dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, '.env.dot')

LOCAL_ENV_LOADED = False
if os.path.exists(env_path) and dotenv.load_dotenv(env_path):
    LOCAL_ENV_LOADED = True
    print("env_path:", env_path)
else:
    LOCAL_ENV_LOADED = False
    print("env_path not found or load failed:", env_path)

def get_env(key, default=None):    
    if not LOCAL_ENV_LOADED:
        return os.environ.get(key, default)
    else:
        return os.environ.get(key) or dotenv.get_key(env_path) or default