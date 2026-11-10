"""
SafeRoute location sharing router.

Add this to your existing backend/api.py:

    from location_share_service import router as location_share_router
    app.include_router(location_share_router)

This keeps location sharing separate from the existing route/safety code.
The store is in memory, which is suitable for the college-project/demo setup.
"""

from datetime import datetime, timezone, timedelta
from secrets import token_urlsafe
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()

# share_id -> session
LIVE_LOCATION_SESSIONS = {}

SESSION_TTL_MINUTES = 120


class LocationUpdate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy: Optional[float] = None


def _now():
    return datetime.now(timezone.utc)


def _cleanup_sessions():
    cutoff = _now() - timedelta(minutes=SESSION_TTL_MINUTES)

    expired = [
        share_id
        for share_id, session in LIVE_LOCATION_SESSIONS.items()
        if session["updated_at"] < cutoff
    ]

    for share_id in expired:
        LIVE_LOCATION_SESSIONS.pop(share_id, None)


@router.post("/share/live/start")
def start_live_location(location: LocationUpdate):
    _cleanup_sessions()

    share_id = token_urlsafe(18)
    now = _now()

    LIVE_LOCATION_SESSIONS[share_id] = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "accuracy": location.accuracy,
        "updated_at": now,
        "active": True
    }

    return {
        "success": True,
        "share_id": share_id,
        "active": True,
        "updated_at": now.isoformat()
    }


@router.post("/share/live/{share_id}")
def update_live_location(
    share_id: str,
    location: LocationUpdate
):
    _cleanup_sessions()

    session = LIVE_LOCATION_SESSIONS.get(share_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Live location session not found or expired."
        )

    if not session["active"]:
        raise HTTPException(
            status_code=410,
            detail="Live location sharing has ended."
        )

    now = _now()

    session.update({
        "latitude": location.latitude,
        "longitude": location.longitude,
        "accuracy": location.accuracy,
        "updated_at": now
    })

    return {
        "success": True,
        "active": True,
        "updated_at": now.isoformat()
    }


@router.get("/share/live/{share_id}")
def get_live_location(share_id: str):
    _cleanup_sessions()

    session = LIVE_LOCATION_SESSIONS.get(share_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Live location session not found or expired."
        )

    return {
        "success": True,
        "share_id": share_id,
        "active": session["active"],
        "latitude": session["latitude"],
        "longitude": session["longitude"],
        "accuracy": session["accuracy"],
        "updated_at": session["updated_at"].isoformat()
    }


@router.post("/share/live/{share_id}/stop")
def stop_live_location(share_id: str):
    session = LIVE_LOCATION_SESSIONS.get(share_id)

    if not session:
        return {
            "success": True,
            "active": False
        }

    session["active"] = False
    session["updated_at"] = _now()

    return {
        "success": True,
        "active": False
    }
