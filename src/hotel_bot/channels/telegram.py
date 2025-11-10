"""
Telegram bot channel implementation with LangChain agent.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
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
            return (
                f"✅ Rezervasyon oluşturuldu!\n\n"
                f"Rezervasyon ID: {res.get('id')}\n"
                f"Oda ID: {res.get('room_id')}\n"
                f"İsim: {res.get('full_name')}\n"
                f"Giriş: {res.get('check_in')}\n"
                f"Çıkış: {res.get('check_out')}\n"
                f"Misafir Sayısı: {res.get('guests')}"
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
        return (
            f"📋 Rezervasyon Detayları:\n\n"
            f"ID: {res.get('id')}\n"
            f"Oda ID: {res.get('room_id')}\n"
            f"İsim: {res.get('full_name')}\n"
            f"Giriş: {res.get('check_in')}\n"
            f"Çıkış: {res.get('check_out')}\n"
            f"Misafir Sayısı: {res.get('guests')}"
        )
    
    return json.dumps(result, indent=2, ensure_ascii=False)


def create_langchain_tools():
    """Create LangChain tools from our tool functions."""
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    # Define input schemas for tools
    class CheckAvailabilityInput(BaseModel):
        check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
        check_out: str = Field(description="Check-out date in YYYY-MM-DD format")
        guests: int = Field(description="Number of guests")
    
    class CreateReservationInput(BaseModel):
        room_id: int = Field(description="Room ID to reserve")
        full_name: str = Field(description="Guest's full name")
        check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
        check_out: str = Field(description="Check-out date in YYYY-MM-DD format")
        guests: int = Field(description="Number of guests")
        phone: Optional[str] = Field(None, description="Phone number (optional)")
        email: Optional[str] = Field(None, description="Email address (optional)")
        notes: Optional[str] = Field(None, description="Additional notes (optional)")
    
    class GetReservationInput(BaseModel):
        reservation_id: int = Field(description="Reservation ID")
    
    class CancelReservationInput(BaseModel):
        reservation_id: int = Field(description="Reservation ID to cancel")
    
    tools = [
        StructuredTool.from_function(
            func=get_room_prices,
            name="get_room_prices",
            description="Get prices for all available rooms. Use this when user asks about room prices or available rooms.",
        ),
        StructuredTool.from_function(
            func=check_availability,
            name="check_availability",
            description="Check room availability for specific dates and number of guests. Use this when user asks about availability or wants to check if rooms are available for specific dates.",
            args_schema=CheckAvailabilityInput,
        ),
        StructuredTool.from_function(
            func=create_reservation,
            name="create_reservation",
            description="Create a new reservation. Use this when user wants to make a reservation. Make sure to get all required information: room_id, full_name, check_in date, check_out date, and number of guests.",
            args_schema=CreateReservationInput,
        ),
        StructuredTool.from_function(
            func=get_reservation,
            name="get_reservation",
            description="Get reservation details by ID. Use this when user asks about a specific reservation or wants to view reservation details.",
            args_schema=GetReservationInput,
        ),
        StructuredTool.from_function(
            func=cancel_reservation,
            name="cancel_reservation",
            description="Cancel a reservation by ID. Use this when user wants to cancel their reservation.",
            args_schema=CancelReservationInput,
        ),
    ]
    
    return tools


def create_langchain_agent():
    """Create LangChain agent with tools using new API."""
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables")
    
    model_name = get_groq_model()
    system_prompt = get_system_prompt()
    
    # Create LLM
    llm = ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.7,
    )
    
    # Create tools
    tools = create_langchain_tools()
    
    # Create agent using new LangChain 1.0 API
    # create_agent returns a compiled graph
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    
    return agent


# Global agent instance
_agent: Optional[Any] = None


def get_agent():
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = create_langchain_agent()
    return _agent


async def handle_message_with_agent(
    user_message: str, context: ContextTypes.DEFAULT_TYPE
) -> str:
    """
    Handle user message using LangChain agent with tools.
    
    Args:
        user_message: User's message
        context: Telegram context
    
    Returns:
        Response message
    """
    try:
        agent = get_agent()
        
        # Run agent in executor to avoid blocking
        # LangChain 1.0 agent is a graph that takes input dict with messages
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.invoke({"messages": [HumanMessage(content=user_message)]})
        )
        
        # Extract response from result
        # LangChain 1.0 returns a dict with messages
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                # Get the last message which should be the assistant's response
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    response = last_message.content
                elif isinstance(last_message, dict):
                    response = last_message.get("content", str(last_message))
                else:
                    response = str(last_message)
            else:
                response = "Üzgünüm, bir yanıt oluşturamadım."
        elif hasattr(result, "content"):
            response = result.content
        else:
            response = str(result)
        
        # Format response if it contains tool results
        if isinstance(response, dict):
            response = json.dumps(response, indent=2, ensure_ascii=False)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in LangChain agent: {e}", exc_info=True)
        return f"❌ Bir hata oluştu: {str(e)}\n\nLütfen tekrar deneyin veya sorunuzu farklı şekilde ifade edin."


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
