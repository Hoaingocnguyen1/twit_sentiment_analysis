from twikit import Client
import os
import dotenv
from typing_extensions import Literal
from ..utils.schemas import TwitterQuery
import json
import warnings
import logging
from twikit._captcha.capsolver import Capsolver


dotenv.load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TwikitClient:
    def __init__(self, language="en-US", captcha_solver=os.getenv("CAPCHA_SOLVER_API")):
        self.is_logged_in = False

        if not captcha_solver:
            warnings.warn(
                "Captcha solver not provided. This might cause your account to be suspended."
            )

        solver = Capsolver(api_key=captcha_solver, max_attempts=5)
        self.client = Client(language, captcha_solver=solver)

    async def login(self, mode: Literal["cookies", "username_password"] = "cookies"):
        """Async method to login to Twitter"""
        if mode == "cookies":
            self.client.load_cookies("cookies.json")
            logging.info("Logged in with cookies")
            self.is_logged_in = True
            return
        else:
            USERNAME = os.getenv("TW_USERNAME")
            EMAIL = os.getenv("TW_EMAIL")
            PASSWORD = os.getenv("TW_PASSWORD")

            await self.client.login(
                auth_info_1=USERNAME, auth_info_2=EMAIL, password=PASSWORD
            )
            self.client.save_cookies("/opt/airflow/cookies.json")
            logging.info("Logged in with username and password")
            self.is_logged_in = True
            return

    async def get_tweets(self, tq: TwitterQuery):
        """Async method to get tweets"""
        if not self.is_logged_in:
            await self.login()

        query_str = f"{tq.query}"
        if tq.since:
            query_str += f" since:{tq.since.date()}"
        if tq.until:
            query_str += f" until:{tq.until.date()}"

        tweets = await self.client.search_tweet(
            query=query_str, product=tq.mode, count=tq.max_results
        )
        return tweets


class TweetCache:
    def __init__(self):
        self.cache = []

    def add(self, tweet_list):
        """Thêm dữ liệu vào cache"""
        self.cache.extend(tweet_list)

    def get_all(self):
        """Lấy toàn bộ dữ liệu từ cache (không clear)"""
        return self.cache

    def save_to_file(self, base_dir, output_file):
        """Ghi dữ liệu cache vào file"""
        os.makedirs(base_dir, exist_ok=True)

        path = os.path.join(base_dir, output_file)

        # Kiểm tra nếu file đã tồn tại, thì append vào, nếu chưa tạo mới
        if os.path.exists(path):
            with open(path, "r+", encoding="utf-8") as f:
                # Đọc dữ liệu hiện có
                existing_data = json.load(f)
                existing_data.extend(self.cache)  # Gắn dữ liệu mới vào cuối

                # Di chuyển con trỏ về đầu file để ghi đè lại
                f.seek(0)
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Appended data to: {path}")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved new data to: {path}")

        # Xóa bộ nhớ cache sau khi lưu vào file
        self.cache = []
