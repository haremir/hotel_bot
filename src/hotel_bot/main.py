"""
Main entry point for the hotel bot.
"""
from __future__ import annotations

import asyncio
import logging

from hotel_bot.channels.telegram import run_telegram_bot

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main():
    """Main function to run the hotel bot."""
    try:
        logger.info("Starting Hotel Bot...")
        asyncio.run(run_telegram_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
