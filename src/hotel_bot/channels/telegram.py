"""
Telegram bot channel implementation with LangChain tool-calling workflow.
"""
from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from hotel_bot.config import get_groq_api_key, get_groq_model, get_telegram_bot_token
from hotel_bot.prompts import get_system_prompt
from hotel_bot.tools import (
    cancel_reservation,
    check_availability,
    create_reservation,
    get_adapter,
    get_reservation,
    get_room_prices,
)

logger = logging.getLogger(__name__)


def format_tool_result(result: Dict[str, Any]) -> str:
    """Format tool result for display."""
    if "error" in result:
        return f"❌ Hata: {result['error']}"
    
    if "success" in result and result["success"]:
        if "reservation" in result:
            res = result["reservation"]
            reference_code = res.get('reference_code') or f"RSV-{res.get('id')}"
            return (
                f"✅ Rezervasyon oluşturuldu!\n\n"
                f"Referans: {reference_code}\n"
                f"Rezervasyon ID: {res.get('id')}\n"
                f"Oda ID: {res.get('room_id')}\n"
                f"İsim: {res.get('full_name')}\n"
                f"Giriş: {res.get('check_in')}\n"
                f"Çıkış: {res.get('check_out')}\n"
                f"Misafir Sayısı: {res.get('guests')}\n"
                f"Telefon: {res.get('phone') or '—'}\n"
                f"E-posta: {res.get('email') or '—'}\n"
                f"Not: {res.get('notes') or '—'}"
            )
        elif "message" in result:
            return f"✅ {result['message']}"
    
    if "rooms" in result:
        rooms = result["rooms"]
        if not rooms:
            return "❌ Şu anda müsait oda bulunmamaktadır."
        lines = ["🏨 Müsait Odalar ve Fiyatlar:\n"]
        for room in rooms:
            lines.append(
                f"• {room['name']} - {room['price_per_night']}₺/gece "
                f"(Kapasite: {room['capacity']} kişi)"
            )
        return "\n".join(lines)
    
    if "available_rooms" in result:
        rooms = result["available_rooms"]
        if not rooms:
            return (
                f"❌ {result.get('check_in')} - {result.get('check_out')} tarihleri arasında "
                f"{result.get('guests')} kişilik müsait oda bulunmamaktadır."
            )
        lines = [
            f"✅ {result.get('check_in')} - {result.get('check_out')} tarihleri arasında "
            f"{result.get('guests')} kişilik müsait odalar:\n"
        ]
        for room in rooms:
            lines.append(
                f"• Oda {room['id']}: {room['name']} - {room['price_per_night']}₺/gece "
                f"(Kapasite: {room['capacity']} kişi)"
            )
        return "\n".join(lines)
    
    if "reservation" in result:
        res = result["reservation"]
        reference_code = res.get('reference_code') or f"RSV-{res.get('id')}"
        return (
            f"📋 Rezervasyon Detayları:\n\n"
            f"Referans: {reference_code}\n"
            f"ID: {res.get('id')}\n"
            f"Oda ID: {res.get('room_id')}\n"
            f"İsim: {res.get('full_name')}\n"
            f"Giriş: {res.get('check_in')}\n"
            f"Çıkış: {res.get('check_out')}\n"
            f"Misafir Sayısı: {res.get('guests')}\n"
            f"Telefon: {res.get('phone') or '—'}\n"
            f"E-posta: {res.get('email') or '—'}\n"
            f"Not: {res.get('notes') or '—'}"
        )
    
    return json.dumps(result, indent=2, ensure_ascii=False)


def create_langchain_tools() -> List[StructuredTool]:
    """Create LangChain StructuredTool objects from our tool functions."""

    tools = [
        StructuredTool.from_function(
            func=get_room_prices,
            name="get_room_prices",
            description="Get prices for all available rooms. Use this when user asks about room prices or wants to see available rooms.",
        ),
        StructuredTool.from_function(
            func=check_availability,
            name="check_availability",
            description="Check room availability for specific dates and number of guests.",
        ),
        StructuredTool.from_function(
            func=create_reservation,
            name="create_reservation",
            description="Create a new reservation. Always ensure you have room_id, guest name, dates, and guest count before calling.",
        ),
        StructuredTool.from_function(
            func=get_reservation,
            name="get_reservation",
            description="Get reservation details by reservation_id.",
        ),
        StructuredTool.from_function(
            func=cancel_reservation,
            name="cancel_reservation",
            description="Cancel an existing reservation by reservation_id.",
        ),
    ]

    return tools


_tools: Optional[List[StructuredTool]] = None
_tool_map: Dict[str, StructuredTool] = {}
_llm: Optional[ChatGroq] = None


def get_tools() -> List[StructuredTool]:
    """Get or create tool instances."""
    global _tools, _tool_map
    if _tools is None:
        _tools = create_langchain_tools()
        _tool_map = {tool.name: tool for tool in _tools}
    return _tools


def get_tool_map() -> Dict[str, StructuredTool]:
    if not _tool_map:
        get_tools()
    return _tool_map


def get_llm() -> ChatGroq:
    """Create or return cached LLM instance."""
    global _llm
    if _llm is None:
        api_key = get_groq_api_key()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        model_name = get_groq_model()
        _llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.4,
        )
    return _llm


def _ensure_database_seeded() -> None:
    """Ensure SQLite adapter has base data."""
    try:
        adapter = get_adapter()
        adapter.insert_sample_rooms_if_empty()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to seed database: %s", exc)


def _prepare_history(context: ContextTypes.DEFAULT_TYPE) -> List[Any]:
    history = context.user_data.get("history")
    if history is None:
        history = [SystemMessage(content=get_system_prompt())]
        context.user_data["history"] = history
    return history


def _trim_history(messages: List[Any], limit: int = 12) -> List[Any]:
    if not messages:
        return []
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    others = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    trimmed = others[-limit:]
    return system_messages[:1] + trimmed


# Seed database once when module is imported
_ensure_database_seeded()


def _run_tool_loop(user_message: str, history: List[Any]) -> tuple[str, List[Any]]:
    """Blocking helper that runs the tool-calling loop."""
    llm = get_llm()
    tools = get_tools()
    tool_map = get_tool_map()
    llm_with_tools = llm.bind_tools(tools)

    messages: List[Any] = list(history)
    human_msg = HumanMessage(content=user_message)
    messages.append(human_msg)
    tool_outputs: List[str] = []

    max_iterations = 4
    for _ in range(max_iterations):
        ai_message: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            content = ai_message.content
            if isinstance(content, list):
                content = " ".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            response_text = str(content).strip()
            if tool_outputs:
                formatted_outputs = "\n\n".join(tool_outputs)
                if formatted_outputs not in response_text:
                    response_text = (
                        f"{response_text}\n\n{formatted_outputs}" if response_text else formatted_outputs
                    )
            return response_text, messages

        tool_messages: List[ToolMessage] = []
        for call in ai_message.tool_calls:
            tool_name = call.get("name")
            tool_call_id = call.get("id")
            args = call.get("args", {})
            tool = tool_map.get(tool_name)
            if not tool:
                tool_messages.append(
                    ToolMessage(
                        content=f"İstenilen {tool_name} aracı bulunamadı.",
                        tool_call_id=tool_call_id,
                        name=tool_name or "unknown_tool",
                    )
                )
                continue

            try:
                raw_result = tool.invoke(args)
                formatted_result = format_tool_result(raw_result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
                formatted_result = f"❌ Araç çalıştırılırken hata oluştu: {exc}"

            tool_outputs.append(formatted_result)
            tool_messages.append(
                ToolMessage(
                    content=formatted_result,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

        messages.extend(tool_messages)

    # If we've exhausted max_iterations, return fallback
    fallback = (
        "İşlem tamamlanamadı. Lütfen isteğinizi daha net bir şekilde tekrarlar mısınız?"
    )
    if tool_outputs:
        formatted_outputs = "\n\n".join(tool_outputs)
        fallback = f"{fallback}\n\n{formatted_outputs}"
    return fallback, messages


async def handle_message_with_agent(
    user_message: str, context: ContextTypes.DEFAULT_TYPE
) -> str:
    """Async wrapper around the blocking tool loop."""
    history = _prepare_history(context)
    history_snapshot = list(history)

    loop = asyncio.get_running_loop()
    response, updated_messages = await loop.run_in_executor(
        None, partial(_run_tool_loop, user_message, history_snapshot)
    )

    context.user_data["history"] = _trim_history(updated_messages)
    return response


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    hotel_name = context.bot_data.get("hotel_name", "Otelimiz")
    welcome_message = (
        f"🏨 Hoş geldiniz! {hotel_name}'e rezervasyon asistanıyım.\n\n"
        f"Size nasıl yardımcı olabilirim?\n\n"
        f"Yapabileceğim işlemler:\n"
        f"• Oda fiyatlarını gösterme\n"
        f"• Müsaitlik kontrolü\n"
        f"• Rezervasyon oluşturma\n"
        f"• Rezervasyon sorgulama\n"
        f"• Rezervasyon iptali\n\n"
        f"Örnek: 'Oda fiyatlarınızı görebilir miyim?' veya "
        f"'25 Aralık için müsait odanız var mı?'"
    )
    await update.message.reply_text(welcome_message)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages."""
    user_message = update.message.text
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    # Process message with agent
    response = await handle_message_with_agent(user_message, context)
    
    # Send response
    await update.message.reply_text(response)


def create_telegram_app() -> Application:
    """Create and configure Telegram application."""
    token = get_telegram_bot_token()
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN environment variable is not set. "
            "Please add it to your .env file."
        )
    
    # Create application
    application = Application.builder().token(token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )
    
    # Store hotel name in bot data
    from hotel_bot.config import get_hotel_name
    application.bot_data["hotel_name"] = get_hotel_name()
    
    return application


async def run_telegram_bot() -> None:
    """Run the Telegram bot."""
    application = create_telegram_app()
    
    logger.info("Starting Telegram bot...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    logger.info("Telegram bot is running. Press Ctrl+C to stop.")
    
    # Keep the bot running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Stopping Telegram bot...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    asyncio.run(run_telegram_bot())