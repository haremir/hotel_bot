# hote_bot
This project is an artificial intelligence-supported reservation system for hotels.

hotel-bot-core/
├── src/hotel_bot/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Ayarlar
│   │
│   ├── llm.py                  # Groq + Llama (fallback)
│   ├── prompts.py              # System promptlar
│   ├── tools.py                # Rezervasyon fonksiyonları
│   │
│   ├── channels/               # ⭐ KANAL KLASÖRÜ
│   │   ├── __init__.py
│   │   ├── telegram.py         # Telegram bot
│   │   ├── whatsapp.py         # WhatsApp (sonra)
│   │   └── instagram.py        # Instagram (sonra)
│   │
│   └── adapters/               # ⭐ VERİTABANI ADAPTERLARI
│       ├── __init__.py
│       ├── base.py             # Abstract base class
│       ├── sqlite_adapter.py   # SQLite için
│       ├── excel_adapter.py    # Excel için
│       └── api_adapter.py      # API'ler için (HotelRunner vb.)
│
├── tests/
├── .env.example
├── pyproject.toml
├── README.md
└── .gitignore

## Quickstart

1) Create and fill your environment variables:

Create a `.env` file based on the keys below and set your credentials (Groq, Telegram, etc.).

Required variables:

```
ENV=development

GROQ_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile
LLM_TIMEOUT=60

TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

DATABASE_URL=sqlite:///hotel_bot.db

HOTEL_NAME=Demo Hotel
HOTEL_PHONE=+90 555 555 55 55
HOTEL_EMAIL=info@demo-hotel.com
```

2) Install dependencies (with uv or pip):

```bash
uv sync
# or
pip install -e .
```

3) Run Telegram bot:

```bash
python -m src.hotel_bot.main
```

Notes:
- LLM is powered by Groq's Llama model. Ensure `GROQ_API_KEY` is valid.
- WhatsApp and Instagram channel stubs are present to be implemented later.