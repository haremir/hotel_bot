
from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_database_url(default: Optional[str] = None) -> str:
    """Get database URL from environment variable."""
    return os.getenv("DATABASE_URL", default or "sqlite:///hotel_bot.db")


def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from environment variable."""
    return os.getenv("GROQ_API_KEY")


def get_groq_model(default: str = "llama-3.1-70b-versatile") -> str:
    """Get Groq model name from environment variable."""
    return os.getenv("GROQ_MODEL", default)


def get_llm_timeout(default: int = 60) -> int:
    """Get LLM timeout in seconds from environment variable."""
    try:
        return int(os.getenv("LLM_TIMEOUT", str(default)))
    except (ValueError, TypeError):
        return default


def get_telegram_bot_token() -> Optional[str]:
    """Get Telegram bot token from environment variable."""
    return os.getenv("TELEGRAM_BOT_TOKEN")


def get_hotel_name(default: str = "Demo Hotel") -> str:
    """Get hotel name from environment variable."""
    return os.getenv("HOTEL_NAME", default)


def get_hotel_phone() -> Optional[str]:
    """Get hotel phone number from environment variable."""
    return os.getenv("HOTEL_PHONE")


def get_hotel_email() -> Optional[str]:
    """Get hotel email from environment variable."""
    return os.getenv("HOTEL_EMAIL")



