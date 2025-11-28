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
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from hotel_bot.config import get_config
from hotel_bot.prompts import get_system_prompt
from hotel_bot.tools import (
    cancel_reservation,
    check_availability,
    create_reservation,
    get_adapter,
    get_reservation,
    get_room_prices,
    update_reservation,
)

logger = logging.getLogger(__name__)

# Timeout settings
TOOL_LOOP_TIMEOUT = 45  # seconds for tool execution
LLM_CALL_TIMEOUT = 30   # seconds for single LLM call


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
            func=update_reservation,
            name="update_reservation",
            description="Update an existing reservation. Only call when user provides reservation ID and fields to update.",
        ),
        StructuredTool.from_function(
            func=get_reservation,
            name="get_reservation",
            description="Get reservation details by reservation_id. Only call when user provides a reservation ID.",
        ),
        StructuredTool.from_function(
            func=cancel_reservation,
            name="cancel_reservation",
            description="Cancel an existing reservation by reservation_id. Only call when user provides a reservation ID.",
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
        config = get_config()
        api_key = config.get_groq_api_key()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        model_name = config.get_groq_model()
        _llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.4,
            timeout=LLM_CALL_TIMEOUT,  # ⭐ Timeout ekledik
            max_retries=2,  # ⭐ Retry ekledik
        )
    return _llm


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


def _run_tool_loop(user_message: str, history: List[Any]) -> tuple[str, List[Any]]:
    """
    Blocking helper that runs the tool-calling loop.
    
    Returns:
        Tuple of (response_text, updated_messages)
    
    Raises:
        TimeoutError: If execution takes too long
        Exception: For other errors
    """
    llm = get_llm()
    tools = get_tools()
    tool_map = get_tool_map()
    llm_with_tools = llm.bind_tools(tools)

    messages: List[Any] = list(history)
    human_msg = HumanMessage(content=user_message)
    messages.append(human_msg)

    max_iterations = 4
    for iteration in range(max_iterations):
        try:
            logger.info(f"Tool loop iteration {iteration + 1}/{max_iterations}")
            
            # Call LLM
            ai_message: AIMessage = llm_with_tools.invoke(messages)
            messages.append(ai_message)

            # Check if we have a final response (no tool calls)
            if not ai_message.tool_calls:
                content = ai_message.content
                if isinstance(content, list):
                    content = " ".join(
                        part["text"]
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                response_text = str(content).strip()
                logger.info("Got final response from LLM")
                return response_text, messages

            # Execute tool calls
            tool_messages: List[ToolMessage] = []
            for call in ai_message.tool_calls:
                tool_name = call.get("name")
                tool_call_id = call.get("id")
                args = call.get("args", {})
                
                logger.info(f"Executing tool: {tool_name} with args: {args}")
                
                tool = tool_map.get(tool_name)
                if not tool:
                    error_msg = f"İstenilen {tool_name} aracı bulunamadı."
                    logger.error(error_msg)
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name or "unknown_tool",
                        )
                    )
                    continue

                try:
                    result = tool.invoke(args)
                    logger.info(f"Tool {tool_name} executed successfully")
                except Exception as exc:
                    logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
                    result = f"❌ Araç çalıştırılırken hata oluştu: {str(exc)}"

                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

            messages.extend(tool_messages)
            
        except Exception as e:
            logger.error(f"Error in tool loop iteration {iteration}: {e}", exc_info=True)
            # Rate limit hatası kontrolü
            if "rate_limit" in str(e).lower() or "429" in str(e):
                raise Exception("API rate limit aşıldı. Lütfen birkaç saniye bekleyip tekrar deneyin.")
            # Timeout hatası
            if "timeout" in str(e).lower():
                raise TimeoutError("İstek zaman aşımına uğradı. Lütfen tekrar deneyin.")
            # Diğer hatalar
            raise

    # If we've exhausted max_iterations, return fallback
    logger.warning("Tool loop exhausted max iterations")
    fallback = (
        "İşlem tamamlanamadı. Lütfen isteğinizi daha net bir şekilde tekrarlar mısınız?"
    )
    return fallback, messages


async def handle_message_with_agent(
    user_message: str, context: ContextTypes.DEFAULT_TYPE
) -> str:
    """
    Async wrapper around the blocking tool loop with timeout.
    
    Returns:
        Response text
        
    Raises:
        TimeoutError: If processing takes too long
        Exception: For other errors
    """
    history = _prepare_history(context)
    history_snapshot = list(history)

    loop = asyncio.get_running_loop()
    
    try:
        # ⭐ Timeout ile çalıştır
        response, updated_messages = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                partial(_run_tool_loop, user_message, history_snapshot)
            ),
            timeout=TOOL_LOOP_TIMEOUT
        )
        
        context.user_data["history"] = _trim_history(updated_messages)
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Tool loop timed out after {TOOL_LOOP_TIMEOUT} seconds")
        raise TimeoutError(
            f"İşlem {TOOL_LOOP_TIMEOUT} saniyeden fazla sürdü. "
            "Lütfen daha basit bir istek ile tekrar deneyin."
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    try:
        hotel_name = context.bot_data.get("hotel_name") or get_config().get_hotel_display_name()
        welcome_message = (
            f"🏨 Hoş geldiniz! {hotel_name}'e rezervasyon asistanıyım.\n\n"
            f"Size nasıl yardımcı olabilirim?\n\n"
            f"Yapabileceğim işlemler:\n"
            f"• Oda fiyatlarını gösterme\n"
            f"• Müsaitlik kontrolü\n"
            f"• Rezervasyon oluşturma\n"
            f"• Rezervasyon güncelleme\n"
            f"• Rezervasyon sorgulama\n"
            f"• Rezervasyon iptali\n\n"
            f"Örnek: 'Oda fiyatlarınızı görebilir miyim?' veya "
            f"'25 Aralık için müsait odanız var mı?'"
        )
        await update.message.reply_text(welcome_message)
    except Exception as e:
        logger.error(f"Error in start_command: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ Bir hata oluştu. Lütfen tekrar /start yazın."
        )


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages with comprehensive error handling."""
    user_message = update.message.text
    
    try:
        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, 
            action="typing"
        )
        
        logger.info(f"Processing message from user {update.effective_user.id}: {user_message[:50]}...")
        
        # Process message with agent
        response = await handle_message_with_agent(user_message, context)
        
        # Kaçış karakterlerini temizle ve formatla
        response = response.replace("\\n", "\n")
        
        # Maksimum 4096 karaktere böl (Telegram limiti)
        if len(response) > 4096:
            chunks = [response[i:i+4096] for i in range(0, len(response), 4096)]
            for chunk in chunks:
                await update.message.reply_text(chunk)
        else:
            await update.message.reply_text(response)
            
        logger.info("Message processed successfully")
        
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        error_message = (
            "⏱️ İşlem çok uzun sürdü.\n\n"
            "Lütfen:\n"
            "• Daha basit bir soru sorun\n"
            "• Birkaç saniye bekleyip tekrar deneyin\n\n"
            "Sorun devam ederse /start ile yeniden başlayın."
        )
        await update.message.reply_text(error_message)
        
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"Error in message_handler: {e}", exc_info=True)
        
        # Spesifik hata mesajları
        if "rate_limit" in error_str or "429" in error_str:
            error_message = (
                "⚠️ API limit aşıldı.\n\n"
                "Lütfen 10-15 saniye bekleyip tekrar deneyin."
            )
        elif "timeout" in error_str:
            error_message = (
                "⏱️ Bağlantı zaman aşımına uğradı.\n\n"
                "Lütfen tekrar deneyin."
            )
        elif "api key" in error_str or "unauthorized" in error_str:
            error_message = (
                "🔑 API anahtarı hatası.\n\n"
                "Lütfen yönetici ile iletişime geçin."
            )
        else:
            error_message = (
                "❌ Beklenmeyen bir hata oluştu.\n\n"
                "Lütfen:\n"
                "• Mesajınızı tekrar gönderin\n"
                "• Sorun devam ederse /start ile yeniden başlayın\n\n"
                f"Hata kodu: {type(e).__name__}"
            )
        
        await update.message.reply_text(error_message)


def create_telegram_app() -> Application:
    """Create and configure Telegram application."""
    config = get_config()
    token = config.get_telegram_bot_token()
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN environment variable is not set. "
            "Please add it to your .env file."
        )
    
    # Create application with timeout settings
    application = (
        Application.builder()
        .token(token)
        .connect_timeout(30)  # ⭐ Connection timeout
        .read_timeout(30)     # ⭐ Read timeout
        .write_timeout(30)    # ⭐ Write timeout
        .build()
    )
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )
    
    # Store hotel name in bot data
    application.bot_data["hotel_name"] = config.get_hotel_display_name()
    
    return application


async def run_telegram_bot() -> None:
    """Run the Telegram bot."""
    application = create_telegram_app()
    
    logger.info("Starting Telegram bot...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        drop_pending_updates=True  # ⭐ Eski mesajları atla
    )
    
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