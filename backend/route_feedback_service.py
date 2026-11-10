import hashlib
import json
import os
import threading
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(BASE_DIR, "datasets", "feedback")
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "route_ratings.json")
_LOCK = threading.Lock()


def _load_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print("Route feedback load error:", exc)
        return {}


def _save_feedback(data):
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    temp_path = FEEDBACK_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, FEEDBACK_PATH)


def make_route_key(source, destination, coordinates):
    """Stable key for a road route. Uses sampled geometry, not route id."""
    coords = coordinates if isinstance(coordinates, list) else []
    if not coords:
        return ""

    sample_count = min(25, len(coords))
    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(i * (len(coords) - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    sampled = []
    for idx in indices:
        point = coords[idx]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        sampled.append([
            round(float(point[0]), 4),
            round(float(point[1]), 4)
        ])

    payload = {
        "source": str(source or "").strip().lower(),
        "destination": str(destination or "").strip().lower(),
        "geometry": sampled
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_route_rating(route_key):
    if not route_key:
        return {"average_rating": 0.0, "rating_count": 0, "rating_score": 0.0}

    with _LOCK:
        data = _load_feedback()
        record = data.get(route_key, {})
        ratings = record.get("ratings", [])
        values = [float(x.get("rating", 0)) for x in ratings if 1 <= float(x.get("rating", 0)) <= 5]

    if not values:
        return {"average_rating": 0.0, "rating_count": 0, "rating_score": 0.0}

    average = sum(values) / len(values)
    return {
        "average_rating": round(average, 2),
        "rating_count": len(values),
        "rating_score": round(average * 20.0, 2),
    }


def save_route_rating(route_key, rating, comment="", source="", destination=""):
    if not route_key:
        raise ValueError("route_key is required")
    rating = float(rating)
    if not 1 <= rating <= 5:
        raise ValueError("rating must be between 1 and 5")

    with _LOCK:
        data = _load_feedback()
        record = data.setdefault(route_key, {"ratings": []})
        record.setdefault("ratings", []).append({
            "rating": rating,
            "comment": str(comment or "").strip()[:500],
            "source": str(source or "").strip()[:200],
            "destination": str(destination or "").strip()[:200],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        _save_feedback(data)

    return get_route_rating(route_key)
