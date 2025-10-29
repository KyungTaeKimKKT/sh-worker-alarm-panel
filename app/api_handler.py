from typing import Optional
import requests

class ApiHandler:
    def __init__(self, base_url:str, **kwargs):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.access_token:Optional[str] = None
        self.refresh_token:Optional[str] = None

        # self.login(login_url, login_info)

    def login(self, login_url:str,login_info:dict) -> bool:
        """ 
        login_url: str = /로 시작해야 함. DRF_LOGIN_URL
        login_info: dict 
        """
        url = f"{self.base_url}{login_url}"
        headers = {"Content-Type": "application/json"}
       
        response = requests.post(url, headers=headers, json=login_info)
        if response.status_code != 200:
            raise Exception(f"Login failed: {response.text}")
        
        self.access_token = response.json().get("access")
        self.refresh_token = response.json().get("refresh")
        return True

    def get_data(self, url:str):
        url = f"{self.base_url}/{url}"
        headers = {"Content-Type": "application/json"}
        response = requests.get(self.get_full_url(url), headers=headers)
        return response.json()

    def get(self, url:str, **kwargs) -> requests.Response:        
        if self.access_token:
            self.headers["Authorization"] = f"JWT {self.access_token}"
        response = requests.get(self.get_full_url(url), headers=self.headers, **kwargs)
        return response


    def get_full_url(self, url:str) -> str:
        return f"{self.base_url}/{url}"