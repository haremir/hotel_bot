
from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_database_url(default: Optional[str] = None) -> str:
    return os.getenv("DATABASE_URL", default or "sqlite:///hotel_bot.db")



