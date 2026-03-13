# a_passwords.py
import os

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

SENDER_EMAIL = _require_env("SENDER_EMAIL")

# 用逗号分隔，自动转成 list
RECIPIENTS = [
    _require_env("RECIPIENTS_str")
]

google_email_app_password = _require_env("GOOGLE_EMAIL_APP_PASSWORD")

TUSHARE_TOKEN = _require_env('TUSHARE_TOKEN') 