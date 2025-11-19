from __future__ import annotations

from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from hotel_bot.adapters.base import ReservationAdapter
from hotel_bot.adapters.sqlite_adapter import SQLiteReservationAdapter
from hotel_bot.config import get_database_url

# Global adapter instance
_adapter: Optional[ReservationAdapter] = None


def _with_reference_code(reservation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure reservation dict contains a human friendly reference code."""
    if not reservation:
        return reservation
    if "reference_code" not in reservation and reservation.get("id") is not None:
        reservation["reference_code"] = f"RSV-{reservation['id']:06d}"
    return reservation


def get_adapter() -> ReservationAdapter:
    """Get or create the database adapter instance."""
    global _adapter
    if _adapter is None:
        db_url = get_database_url()
        _adapter = SQLiteReservationAdapter(db_url)
        _adapter.init()
    return _adapter


def set_adapter(adapter: ReservationAdapter) -> None:
    """Set a custom adapter instance (useful for testing)."""
    global _adapter
    _adapter = adapter


def tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool."""
    func._is_tool = True
    func._tool_name = func.__name__
    func._tool_description = func.__doc__ or ""
    return func


def _dates_overlap(
    check_in1: str, check_out1: str, check_in2: str, check_out2: str
) -> bool:
    """Check if two date ranges overlap."""
    d1_in = datetime.strptime(check_in1, "%Y-%m-%d")
    d1_out = datetime.strptime(check_out1, "%Y-%m-%d")
    d2_in = datetime.strptime(check_in2, "%Y-%m-%d")
    d2_out = datetime.strptime(check_out2, "%Y-%m-%d")
    
    # Overlap if: d1_in < d2_out and d2_in < d1_out
    return d1_in < d2_out and d2_in < d1_out


@tool
def get_room_prices() -> Dict[str, Any]:
    """
    Get prices for all available rooms.
    
    Returns:
        Dict with 'rooms' list containing room information with prices.
    """
    adapter = get_adapter()
    rooms = adapter.list_rooms(only_available=True)
    
    return {
        "rooms": [
            {
                "id": room["id"],
                "name": room["name"],
                "capacity": room["capacity"],
                "price_per_night": room["price_per_night"],
                "status": room["status"],
            }
            for room in rooms
        ]
    }


@tool
def check_availability(
    check_in: str, check_out: str, guests: int
) -> Dict[str, Any]:
    """
    Check room availability for given dates and number of guests.
    
    Args:
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests
    
    Returns:
        Dict with 'available_rooms' list containing available rooms.
    """
    adapter = get_adapter()
    
    # Get all available rooms that can accommodate the guests
    all_rooms = adapter.list_rooms(only_available=True)
    suitable_rooms = [r for r in all_rooms if r["capacity"] >= guests]
    
    # Check for date conflicts
    available_rooms = []
    for room in suitable_rooms:
        reservations = adapter.list_reservations_for_room(room["id"])
        
        # Check if any reservation overlaps with requested dates
        has_conflict = False
        for res in reservations:
            if _dates_overlap(check_in, check_out, res["check_in"], res["check_out"]):
                has_conflict = True
                break
        
        if not has_conflict:
            available_rooms.append({
                "id": room["id"],
                "name": room["name"],
                "capacity": room["capacity"],
                "price_per_night": room["price_per_night"],
            })
    
    return {
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "available_rooms": available_rooms,
    }


@tool
def create_reservation(
    room_id: int,
    full_name: str,
    check_in: str,
    check_out: str,
    guests: int,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new reservation.
    
    Args:
        room_id: ID of the room to reserve
        full_name: Full name of the guest
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests
        phone: Optional phone number
        email: Optional email address
        notes: Optional notes
    
    Returns:
        Dict with reservation details including 'id'.
    """
    adapter = get_adapter()

    if not phone and not email:
        return {
            "error": (
                "Rezervasyon oluşturmak için en az bir iletişim bilgisi (telefon veya e-posta) sağlamalısınız."
            )
        }
    
    # Verify room exists and is available
    room = adapter.get_room(room_id)
    if not room:
        return {"error": f"Room with ID {room_id} not found"}
    
    if room["status"] != "available":
        return {"error": f"Room {room_id} is not available (status: {room['status']})"}
    
    # Check capacity
    if room["capacity"] < guests:
        return {
            "error": f"Room {room_id} can only accommodate {room['capacity']} guests, but {guests} requested"
        }
    
    # Check for date conflicts
    reservations = adapter.list_reservations_for_room(room_id)
    for res in reservations:
        if _dates_overlap(check_in, check_out, res["check_in"], res["check_out"]):
            return {
                "error": f"Room {room_id} is already reserved for the requested dates"
            }
    
    # Create reservation
    reservation = adapter.create_reservation(
        room_id=room_id,
        full_name=full_name,
        check_in=check_in,
        check_out=check_out,
        guests=guests,
        phone=phone,
        email=email,
        notes=notes,
    )
    _with_reference_code(reservation)
    
    return {
        "success": True,
        "reservation": reservation,
    }


@tool
def get_reservation(reservation_id: int) -> Dict[str, Any]:
    """
    Get reservation details by ID.
    
    Args:
        reservation_id: ID of the reservation
    
    Returns:
        Dict with reservation details or error message.
    """
    adapter = get_adapter()
    reservation = adapter.get_reservation(reservation_id)
    _with_reference_code(reservation)
    
    if not reservation:
        return {"error": f"Reservation with ID {reservation_id} not found"}
    
    return {"reservation": reservation}


@tool
def cancel_reservation(reservation_id: int) -> Dict[str, Any]:
    """
    Cancel a reservation by ID.
    
    Args:
        reservation_id: ID of the reservation to cancel
    
    Returns:
        Dict with success status or error message.
    """
    adapter = get_adapter()
    
    # Verify reservation exists
    reservation = adapter.get_reservation(reservation_id)
    if not reservation:
        return {"error": f"Reservation with ID {reservation_id} not found"}
    
    # Delete reservation
    success = adapter.delete_reservation(reservation_id)
    
    if success:
        return {
            "success": True,
            "message": f"Reservation {reservation_id} has been cancelled",
        }
    else:
        return {"error": f"Failed to cancel reservation {reservation_id}"}
