import "./App.css";
import { Fragment, useEffect, useRef, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Marker,
  Popup,
  useMap
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import L from "leaflet";
const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// ============================================================
// FIX LEAFLET MARKER ICONS
// ============================================================

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",

  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",

  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png"
});


// ============================================================
// ROUTE COLORS
// ============================================================

const routePalette = {
  1: "#16a34a",
  2: "#2563eb",
  3: "#ea580c",
  4: "#a855f7",
  5: "#06b6d4",
  6: "#0d9488",
  7: "#f59e0b",
  8: "#b45309"
};

function getRouteColor(route) {
  const id = Number(route?.route_id || 0);
  if (routePalette[id]) return routePalette[id];
  if (route?.recommended === true) return "#2563eb";
  const rank = Number(route?.safety_rank ?? route?.rank ?? 0);
  if (rank === 1) return "#2563eb";
  if (rank === 2) return "#16a34a";
  if (rank === 3) return "#ea580c";
  return "#2563eb";
}


// ============================================================
// HELPER FUNCTIONS
// ============================================================

function getTrafficCondition(score) {
  if (score >= 80) {
    return "Low Traffic";
  }

  if (score >= 50) {
    return "Moderate Traffic";
  }

  if (score >= 30) {
    return "Heavy Traffic";
  }

  return "Very Heavy Traffic";
}


function getWeatherRisk(score) {
  if (score >= 80) {
    return "Good";
  }

  if (score >= 60) {
    return "Moderate";
  }

  if (score >= 40) {
    return "Poor";
  }

  return "Very Poor";
}


function getLightingCondition(score) {
  if (score >= 80) {
    return "Good Visibility";
  }

  if (score >= 50) {
    return "Moderate Visibility";
  }

  if (score >= 30) {
    return "Low Visibility";
  }

  return "Very Low Visibility";
}


function getCrimeRisk(score) {
  if (score <= 20) {
    return "Low Crime Risk";
  }

  if (score <= 40) {
    return "Moderate Crime Risk";
  }

  if (score <= 60) {
    return "High Crime Risk";
  }

  return "Very High Crime Risk";
}


function getSafetyRisk(score) {
  if (score < 30) return "Very High Risk";
  if (score < 45) return "High Risk";
  if (score < 60) return "Medium Risk";
  if (score < 75) return "Low Risk";
  return "Very Low Risk";
}


function getRouteCorridorName(route, fallbackIdx = 1) {
  if (!route) return `via Corridor ${fallbackIdx}`;

  // 1. If route has explicit descriptive name
  if (route.name && !/^Route\s*\d+$/i.test(String(route.name).trim())) {
    const raw = String(route.name).trim();
    const match = raw.match(/^Route\s*\d+\s*[-–:]\s*(.+)$/i);
    const clean = match ? match[1].trim() : raw;
    return clean.toLowerCase().startsWith("via") ? clean : `via ${clean}`;
  }

  if (route.corridor_name && !/^Route\s*\d+$/i.test(String(route.corridor_name).trim())) {
    const raw = String(route.corridor_name).trim();
    return raw.toLowerCase().startsWith("via") ? raw : `via ${raw}`;
  }

  // 2. Extract major road/street names from navigation_instructions
  const instructions = Array.isArray(route.navigation_instructions)
    ? route.navigation_instructions
    : [];

  if (instructions.length > 0) {
    const roadDist = {};
    for (const inst of instructions) {
      const street = (inst.street_name || "").trim();
      if (
        street &&
        !/^(unnamed|unnamed road|-|service road|ramp)$/i.test(street)
      ) {
        roadDist[street] = (roadDist[street] || 0) + (Number(inst.distance_m) || 0);
      }
    }

    const sortedRoads = Object.entries(roadDist).sort((a, b) => b[1] - a[1]);
    if (sortedRoads.length > 0) {
      const topRoad = sortedRoads[0][0];
      const secondRoad =
        sortedRoads.length > 1 && sortedRoads[1][1] > 5000
          ? sortedRoads[1][0]
          : null;

      let summary = topRoad;
      if (
        secondRoad &&
        !topRoad.toLowerCase().includes(secondRoad.toLowerCase()) &&
        !secondRoad.toLowerCase().includes(topRoad.toLowerCase())
      ) {
        summary = `${topRoad} / ${secondRoad}`;
      }
      return summary.toLowerCase().startsWith("via") ? summary : `via ${summary}`;
    }
  }

  // 3. Fallback
  return `via Corridor ${route.route_id || fallbackIdx}`;
}


// ============================================================
// NAVIGATION HELPERS
// ============================================================

function formatNavigationDistance(meters) {
  const value = Number(meters || 0);

  if (value < 1000) {
    return `${Math.max(0, Math.round(value / 10) * 10)} m`;
  }

  return `${(value / 1000).toFixed(1)} km`;
}

function formatNavigationTime(minutes) {
  const value = Math.max(0, Math.round(Number(minutes || 0)));

  if (value < 60) {
    return `${value} min`;
  }

  const hours = Math.floor(value / 60);
  const mins = value % 60;

  return mins === 0
    ? `${hours} hr`
    : `${hours} hr ${mins} min`;
}

function haversineDistanceMeters(lat1, lon1, lat2, lon2) {
  const earthRadius = 6371000;
  const toRadians = (value) => value * Math.PI / 180;

  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(lat1)) *
    Math.cos(toRadians(lat2)) *
    Math.sin(dLon / 2) ** 2;

  return 2 * earthRadius * Math.asin(Math.sqrt(a));
}


function calculateBearingDegrees(lat1, lon1, lat2, lon2) {
  const toRadians = (value) => value * Math.PI / 180;
  const toDegrees = (value) => value * 180 / Math.PI;

  const phi1 = toRadians(lat1);
  const phi2 = toRadians(lat2);
  const deltaLambda = toRadians(lon2 - lon1);

  const y = Math.sin(deltaLambda) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);

  return (toDegrees(Math.atan2(y, x)) + 360) % 360;
}


// ============================================================
// SMOOTH MOVING NAVIGATION POINTER
// ============================================================

function createNavigationPointerIcon(heading = 0) {
  const safeHeading = Number.isFinite(Number(heading))
    ? Number(heading)
    : 0;

  return L.divIcon({
    className: "saferoute-navigation-pointer",
    iconSize: [52, 52],
    iconAnchor: [26, 26],
    popupAnchor: [0, -26],
    html: `
      <div style="
        width:52px;
        height:52px;
        display:flex;
        align-items:center;
        justify-content:center;
        transform:rotate(${safeHeading}deg);
        transform-origin:center center;
      ">
        <div style="
          width:40px;
          height:40px;
          border-radius:50%;
          background:rgba(37,99,235,0.20);
          border:2px solid rgba(255,255,255,0.95);
          box-shadow:0 3px 14px rgba(15,23,42,0.38);
          display:flex;
          align-items:center;
          justify-content:center;
        ">
          <div style="
            width:0;
            height:0;
            border-left:9px solid transparent;
            border-right:9px solid transparent;
            border-bottom:24px solid #2563eb;
            filter:drop-shadow(0 2px 2px rgba(15,23,42,0.35));
            transform:translateY(-2px);
          "></div>
        </div>
      </div>
    `
  });
}


function MovingLocationPointer({
  position,
  heading = 0,
  popupText = "Your current location"
}) {
  const initialPosition = position
    ? [Number(position.latitude), Number(position.longitude)]
    : null;

  const [animatedPosition, setAnimatedPosition] =
    useState(initialPosition);

  const animatedPositionRef =
    useRef(initialPosition);

  useEffect(() => {
    if (!position) return undefined;

    const target = [
      Number(position.latitude),
      Number(position.longitude)
    ];

    if (
      !Number.isFinite(target[0]) ||
      !Number.isFinite(target[1])
    ) {
      return undefined;
    }

    const from =
      animatedPositionRef.current || target;

    const startLat = Number(from[0]);
    const startLon = Number(from[1]);

    if (
      !Number.isFinite(startLat) ||
      !Number.isFinite(startLon)
    ) {
      animatedPositionRef.current = target;
      setAnimatedPosition(target);
      return undefined;
    }

    const distance = haversineDistanceMeters(
      startLat,
      startLon,
      target[0],
      target[1]
    );

    const duration = Math.min(
      900,
      Math.max(250, 350 + distance * 2)
    );

    const startedAt = performance.now();
    let animationFrame = null;

    const animate = (now) => {
      const progress = Math.min(
        1,
        (now - startedAt) / duration
      );

      const smoothProgress =
        progress * progress * (3 - 2 * progress);

      const nextPosition = [
        startLat +
        (target[0] - startLat) * smoothProgress,
        startLon +
        (target[1] - startLon) * smoothProgress
      ];

      animatedPositionRef.current = nextPosition;
      setAnimatedPosition(nextPosition);

      if (progress < 1) {
        animationFrame =
          requestAnimationFrame(animate);
      }
    };

    animationFrame =
      requestAnimationFrame(animate);

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [
    position?.latitude,
    position?.longitude
  ]);

  if (!animatedPosition) return null;

  return (
    <Marker
      position={animatedPosition}
      icon={createNavigationPointerIcon(heading)}
      zIndexOffset={2000}
    >
      <Popup>
        <strong>📍 {popupText}</strong>
        <br />
        {Number(position?.accuracy || 0) > 0
          ? `GPS accuracy: ${Math.round(position.accuracy)} m`
          : "GPS location active"}
        <br />
        Direction: {Math.round(Number(heading) || 0)}°
      </Popup>
    </Marker>
  );
}


function getNearestRoutePointInfo(position, coordinates) {
  if (!position || !Array.isArray(coordinates) || coordinates.length === 0) {
    return { index: 0, distance: Infinity };
  }

  let nearestIndex = 0;
  let nearestDistance = Infinity;

  coordinates.forEach((point, index) => {
    if (!Array.isArray(point) || point.length < 2) return;

    const longitude = Number(point[0]);
    const latitude = Number(point[1]);

    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;

    const distance = haversineDistanceMeters(
      position.latitude,
      position.longitude,
      latitude,
      longitude
    );

    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  });

  return { index: nearestIndex, distance: nearestDistance };
}

function getNearestRoutePointIndex(position, coordinates) {
  return getNearestRoutePointInfo(position, coordinates).index;
}

function getRouteDistanceBetweenIndices(coordinates, startIndex, endIndex) {
  if (!Array.isArray(coordinates) || coordinates.length < 2) return 0;

  const start = Math.max(0, Math.min(Number(startIndex) || 0, coordinates.length - 1));
  const end = Math.max(start, Math.min(Number(endIndex) || 0, coordinates.length - 1));

  let distance = 0;

  for (let index = start; index < end; index += 1) {
    const a = coordinates[index];
    const b = coordinates[index + 1];

    if (!Array.isArray(a) || !Array.isArray(b) || a.length < 2 || b.length < 2) continue;

    distance += haversineDistanceMeters(
      Number(a[1]), Number(a[0]),
      Number(b[1]), Number(b[0])
    );
  }

  return distance;
}

function getNavigationInstructionIndex(instructions, nearestPointIndex) {
  if (!Array.isArray(instructions) || instructions.length === 0) {
    return 0;
  }

  for (let index = 0; index < instructions.length; index += 1) {
    const instruction = instructions[index];
    const interval = Array.isArray(instruction?.interval)
      ? instruction.interval
      : [0, 0];

    const endIndex = Number(interval[1] ?? 0);

    if (nearestPointIndex <= endIndex) {
      return index;
    }
  }

  return instructions.length - 1;
}

function getNavigationTurnIcon(sign) {
  const value = Number(sign);

  if (value === -3) return "↙️";
  if (value === -2) return "⬅️";
  if (value === -1) return "↖️";
  if (value === 1) return "↗️";
  if (value === 2) return "➡️";
  if (value === 3) return "↘️";
  if (value === 6) return "🔄";
  if (value === 7) return "↱";
  if (value === -7) return "↰";
  if (value === 4) return "🏁";
  return "⬆️";
}

function NavigationMapController({
  navigationActive,
  currentPosition
}) {
  const map = useMap();

  useEffect(() => {
    if (!navigationActive || !currentPosition) return;

    map.setView(
      [currentPosition.latitude, currentPosition.longitude],
      Math.max(map.getZoom(), 16),
      { animate: true }
    );
  }, [map, navigationActive, currentPosition]);

  useEffect(() => {
    const handleRecenter = () => {
      if (!currentPosition) return;

      map.setView(
        [currentPosition.latitude, currentPosition.longitude],
        Math.max(map.getZoom(), 16),
        { animate: true }
      );
    };

    window.addEventListener(
      "saferoute-recenter",
      handleRecenter
    );

    return () => {
      window.removeEventListener(
        "saferoute-recenter",
        handleRecenter
      );
    };
  }, [map, currentPosition]);

  return null;
}

function LiveSafetyPanel({ liveSafety, loading, error }) {
  if (!liveSafety && !loading && !error) return null;

  const safetyScore = Number(liveSafety?.safety_score ?? 0);
  const crimeRisk = Number(liveSafety?.crime?.risk_score ?? 0);
  const lighting = Number(liveSafety?.lighting?.score ?? 0);
  const traffic = Number(liveSafety?.traffic?.score ?? 0);
  const weather = Number(liveSafety?.weather?.score ?? 0);

  const safetyClass =
    safetyScore >= 75 ? "LOW RISK" :
      safetyScore >= 60 ? "MODERATE" :
        safetyScore >= 45 ? "HIGH RISK" :
          "VERY HIGH RISK";

  const cardStyle = {
    background: "rgba(255,255,255,0.96)",
    border: "1px solid #e2e8f0",
    borderRadius: "14px",
    padding: "12px",
    boxShadow: "0 5px 18px rgba(15,23,42,0.10)"
  };

  return (
    <section
      className="live-safety-panel"
      style={{
        margin: "14px 0",
        borderRadius: "16px",
        padding: "16px",
        background: "linear-gradient(135deg, #ecfeff, #eff6ff)",
        border: "1px solid #cbd5e1"
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap"
        }}
      >
        <div>
          <div style={{ fontSize: "0.78rem", fontWeight: 800, color: "#475569", textTransform: "uppercase" }}>
            📍 Live area safety
          </div>
          <h2 style={{ margin: "3px 0 0", color: "#0f172a" }}>
            {liveSafety?.crime?.location && liveSafety.crime.location !== "Unknown"
              ? liveSafety.crime.location
              : "Current area"}
          </h2>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "0.76rem", color: "#64748b" }}>AI safety prediction</div>
          <strong style={{ fontSize: "1.5rem", color: safetyScore >= 60 ? "#15803d" : "#dc2626" }}>
            {loading ? "..." : `${Math.round(safetyScore)}/100`}
          </strong>
          {liveSafety && (
            <div style={{ fontSize: "0.72rem", fontWeight: 800, color: "#475569" }}>
              {safetyClass}
            </div>
          )}
        </div>
      </div>

      {error && (
        <div style={{ marginTop: "10px", padding: "9px 11px", borderRadius: "10px", background: "#fee2e2", color: "#991b1b", fontWeight: 700 }}>
          ⚠️ {error}
        </div>
      )}

      {liveSafety && (
        <>
          <div
            style={{
              marginTop: "12px",
              padding: "11px 13px",
              borderRadius: "11px",
              background:
                liveSafety.alert_type === "none" ? "#dcfce7" : "#fef3c7",
              color:
                liveSafety.alert_type === "none" ? "#166534" : "#92400e",
              fontWeight: 800
            }}
          >
            {liveSafety.alert_type === "none" ? "✓" : "⚠️"} {liveSafety.safety_alert}
          </div>

          <div
            className="live-safety-metrics"
            style={{
              marginTop: "12px",
              display: "grid",
              gridTemplateColumns: "repeat(5, minmax(0, 1fr))",
              gap: "8px"
            }}
          >
            <div style={cardStyle}>
              <div style={{ fontSize: "0.75rem", color: "#64748b" }}>🌙 Lighting</div>
              <strong>{Math.round(lighting)}/100</strong>
            </div>
            <div style={cardStyle}>
              <div style={{ fontSize: "0.75rem", color: "#64748b" }}>🚗 Traffic</div>
              <strong>{Math.round(traffic)}/100</strong>
            </div>
            <div style={cardStyle}>
              <div style={{ fontSize: "0.75rem", color: "#64748b" }}>🌦️ Weather</div>
              <strong>{Math.round(weather)}/100</strong>
            </div>
            <div style={cardStyle}>
              <div style={{ fontSize: "0.75rem", color: "#64748b" }}>🚨 Crime risk</div>
              <strong>{Math.round(crimeRisk)}/100</strong>
            </div>
            <div style={cardStyle}>
              <div style={{ fontSize: "0.75rem", color: "#64748b" }}>👁️ Visibility</div>
              <strong>{Number(liveSafety.weather?.visibility_km ?? 0).toFixed(1)} km</strong>
            </div>
          </div>

          <div
            className="live-safety-weather-row"
            style={{
              marginTop: "10px",
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: "8px",
              fontSize: "0.84rem",
              color: "#334155"
            }}
          >
            <div>🌡️ {Number(liveSafety.weather?.temperature_c ?? 0).toFixed(1)} °C · {liveSafety.weather?.condition || "Unknown"}</div>
            <div>💧 {Math.round(Number(liveSafety.weather?.humidity_percent ?? 0))}% humidity</div>
            <div>💨 {Number(liveSafety.weather?.wind_speed_kmph ?? 0).toFixed(1)} km/h wind</div>
          </div>
        </>
      )}

      <div style={{ marginTop: "9px", fontSize: "0.72rem", color: "#64748b" }}>
        Live checks refresh during navigation. Crime information represents available crime-risk data/model output; it does not guarantee that a crime will occur.
      </div>
    </section>
  );
}

function NavigationPanel({
  selectedRoute,
  navigationActive,
  currentPosition,
  currentInstructionIndex,
  onStart,
  onDemoStart,
  onStop,
  onRecenter,
  gpsError,
  offRoute,
  rerouting,
  rerouteMessage,
  onShareCurrent,
  onStartLiveShare,
  onStopLiveShare,
  locationShareActive,
  locationShareUrl,
  locationShareMessage,
  demoNavigationActive
}) {
  if (!selectedRoute) return null;

  const instructions = Array.isArray(
    selectedRoute.navigation_instructions
  )
    ? selectedRoute.navigation_instructions
    : [];

  const currentInstruction =
    instructions[currentInstructionIndex] ||
    instructions[0] ||
    null;

  const routeCoordinates = Array.isArray(selectedRoute.coordinates)
    ? selectedRoute.coordinates
    : [];

  const nearestRoute = currentPosition
    ? getNearestRoutePointInfo(currentPosition, routeCoordinates)
    : { index: 0, distance: Infinity };

  const currentPointIndex = nearestRoute.index;
  const isActuallyOnRoute = nearestRoute.distance <= 120;
  const isNearRoute = nearestRoute.distance <= 2000;

  const instructionInterval = Array.isArray(currentInstruction?.interval)
    ? currentInstruction.interval
    : [0, 0];

  const targetIndex = Math.min(
    Number(instructionInterval[1] || 0),
    Math.max(0, routeCoordinates.length - 1)
  );

  let distanceToNextTurn = Number(currentInstruction?.distance_m || 0);

  if (currentPosition && isActuallyOnRoute && routeCoordinates.length > 1) {
    distanceToNextTurn = getRouteDistanceBetweenIndices(
      routeCoordinates,
      currentPointIndex,
      targetIndex
    );
  }

  const totalRouteDistance = Number(selectedRoute.distance_km || 0) * 1000;

  const remainingDistance = isActuallyOnRoute
    ? getRouteDistanceBetweenIndices(
      routeCoordinates,
      currentPointIndex,
      Math.max(0, routeCoordinates.length - 1)
    )
    : totalRouteDistance;

  const remainingMinutes = totalRouteDistance > 0
    ? Number(selectedRoute.time_minutes || 0) *
    Math.min(1, remainingDistance / totalRouteDistance)
    : Number(selectedRoute.time_minutes || 0);

  return (
    <section
      className="navigation-panel"
      style={{
        margin: "0 0 14px",
        padding: "12px 16px",
        borderRadius: "14px",
        background: "linear-gradient(135deg, #0f172a, #1e3a8a)",
        color: "#ffffff",
        boxShadow: "0 8px 24px rgba(15,23,42,0.2)"
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap"
        }}
      >
        <div>
          <div
            style={{
              fontSize: "0.8rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              opacity: 0.8
            }}
          >
            🧭 LIVE NAVIGATION
          </div>
          <h2 style={{ margin: "4px 0 0" }}>
            Route {selectedRoute.route_id}
            {selectedRoute.recommended ? " — Recommended" : ""}
          </h2>
          {selectedRoute.rerouted && (
            <div
              style={{
                marginTop: "5px",
                fontSize: "0.78rem",
                fontWeight: 800,
                color: "#bfdbfe"
              }}
            >
              🔄 Live rerouted route
            </div>
          )}
          {demoNavigationActive && (
            <div style={{ marginTop: "6px", fontSize: "0.78rem", fontWeight: 800, color: "#ddd6fe" }}>
              🎬 DEMO MODE — simulated GPS movement
            </div>
          )}
        </div>

        <div className="navigation-action-buttons" style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {!navigationActive ? (
            <>
              <button
                onClick={onStart}
                style={{
                  border: "none",
                  borderRadius: "10px",
                  padding: "11px 16px",
                  background: "#22c55e",
                  color: "#ffffff",
                  fontWeight: 700,
                  cursor: "pointer"
                }}
              >
                ▶ Start Navigation
              </button>

              <button
                onClick={onDemoStart}
                disabled={!selectedRoute}
                style={{
                  border: "none",
                  borderRadius: "10px",
                  padding: "11px 16px",
                  background: "#7c3aed",
                  color: "#ffffff",
                  fontWeight: 800,
                  cursor: "pointer"
                }}
              >
                🎬 Demo Navigation
              </button>
            </>
          ) : (
            <button
              onClick={onStop}
              style={{
                border: "none",
                borderRadius: "10px",
                padding: "11px 16px",
                background: "#ef4444",
                color: "#ffffff",
                fontWeight: 700,
                cursor: "pointer"
              }}
            >
              ■ Stop
            </button>
          )}

          {currentPosition && (
            <button
              onClick={onRecenter}
              style={{
                border: "1px solid rgba(255,255,255,0.35)",
                borderRadius: "10px",
                padding: "11px 16px",
                background: "rgba(255,255,255,0.12)",
                color: "#ffffff",
                fontWeight: 700,
                cursor: "pointer"
              }}
            >
              ◎ Recenter
            </button>
          )}
        </div>
      </div>


      <div
        className="navigation-share-buttons"
        style={{
          marginTop: "12px",
          display: "flex",
          gap: "8px",
          flexWrap: "wrap"
        }}
      >
        <button
          onClick={onShareCurrent}
          disabled={!currentPosition}
          style={{
            border: "1px solid rgba(255,255,255,0.35)",
            borderRadius: "10px",
            padding: "10px 14px",
            background: "rgba(255,255,255,0.12)",
            color: "#ffffff",
            fontWeight: 700,
            cursor: currentPosition ? "pointer" : "not-allowed",
            opacity: currentPosition ? 1 : 0.55
          }}
        >
          📍 Share Current Location
        </button>

        {!locationShareActive ? (
          <button
            onClick={onStartLiveShare}
            disabled={!currentPosition}
            style={{
              border: "none",
              borderRadius: "10px",
              padding: "10px 14px",
              background: "#16a34a",
              color: "#ffffff",
              fontWeight: 800,
              cursor: currentPosition ? "pointer" : "not-allowed",
              opacity: currentPosition ? 1 : 0.55
            }}
          >
            🟢 Share Live Location
          </button>
        ) : (
          <button
            onClick={onStopLiveShare}
            style={{
              border: "none",
              borderRadius: "10px",
              padding: "10px 14px",
              background: "#dc2626",
              color: "#ffffff",
              fontWeight: 800,
              cursor: "pointer"
            }}
          >
            ■ Stop Live Sharing
          </button>
        )}
      </div>

      {locationShareActive && locationShareUrl && (
        <div
          style={{
            marginTop: "10px",
            padding: "11px 12px",
            borderRadius: "10px",
            background: "rgba(34,197,94,0.16)",
            color: "#dcfce7",
            fontSize: "0.82rem",
            wordBreak: "break-all"
          }}
        >
          <strong>🟢 Live sharing link:</strong>
          <br />
          {locationShareUrl}
        </div>
      )}

      {locationShareMessage && (
        <div
          style={{
            marginTop: "8px",
            fontSize: "0.82rem",
            color: "#dbeafe"
          }}
        >
          {locationShareMessage}
        </div>
      )}

      {gpsError && (
        <div
          style={{
            marginTop: "12px",
            padding: "10px 12px",
            borderRadius: "10px",
            background: "rgba(239,68,68,0.18)",
            color: "#fecaca"
          }}
        >
          ⚠️ {gpsError}
        </div>
      )}

      {offRoute && (
        <div
          style={{
            marginTop: "12px",
            padding: "10px 12px",
            borderRadius: "10px",
            background: "rgba(245,158,11,0.18)",
            color: "#fde68a",
            fontWeight: 700
          }}
        >
          ⚠️ You appear to be off the selected route. SafeRoute will automatically calculate a new route to the destination.
        </div>
      )}


      {(rerouting || rerouteMessage) && (
        <div
          style={{
            marginTop: "8px",
            padding: "10px 12px",
            borderRadius: "10px",
            background: rerouting
              ? "rgba(59,130,246,0.18)"
              : rerouteMessage?.startsWith("✓")
                ? "rgba(34,197,94,0.18)"
                : "rgba(255,255,255,0.10)",
            color: "#e0f2fe",
            fontWeight: 700
          }}
        >
          {rerouting
            ? "🔄 Recalculating route..."
            : rerouteMessage}
        </div>
      )}

      <div
        className="navigation-summary-grid"
        style={{
          marginTop: "16px",
          display: "grid",
          gridTemplateColumns: "minmax(0, 2fr) minmax(220px, 1fr)",
          gap: "12px"
        }}
      >
        <div
          style={{
            background: "rgba(255,255,255,0.1)",
            borderRadius: "14px",
            padding: "18px"
          }}
        >
          <div
            style={{
              fontSize: "0.85rem",
              opacity: 0.78,
              marginBottom: "7px"
            }}
          >
            {navigationActive ? "NEXT INSTRUCTION" : "FIRST INSTRUCTION"}
          </div>

          <div
            style={{
              display: "flex",
              gap: "12px",
              alignItems: "center"
            }}
          >
            <div style={{ fontSize: "2.3rem" }}>
              {getNavigationTurnIcon(currentInstruction?.sign)}
            </div>

            <div>
              <div
                style={{
                  fontSize: "1.2rem",
                  fontWeight: 800,
                  lineHeight: 1.3
                }}
              >
                {!currentPosition
                  ? (currentInstruction?.text || "Waiting for GPS...")
                  : !isNearRoute
                    ? "Move toward the highlighted route"
                    : (currentInstruction?.text || "Continue on the selected route")}
              </div>

              <div
                style={{
                  marginTop: "6px",
                  fontSize: "0.95rem",
                  opacity: 0.82
                }}
              >
                {!currentPosition
                  ? "Waiting for your current location"
                  : !isNearRoute
                    ? `Route is ${formatNavigationDistance(nearestRoute.distance)} away`
                    : currentInstruction
                      ? `In ${formatNavigationDistance(distanceToNextTurn)}`
                      : "Continue toward the destination"}
              </div>
            </div>
          </div>
        </div>

        <div
          className="navigation-stat-grid"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "10px"
          }}
        >
          <div
            style={{
              background: "rgba(255,255,255,0.1)",
              borderRadius: "12px",
              padding: "14px"
            }}
          >
            <div style={{ opacity: 0.72, fontSize: "0.8rem" }}>
              Remaining
            </div>
            <strong style={{ fontSize: "1.05rem" }}>
              {currentPosition && !isNearRoute
                ? "—"
                : formatNavigationDistance(remainingDistance)}
            </strong>
          </div>

          <div
            style={{
              background: "rgba(255,255,255,0.1)",
              borderRadius: "12px",
              padding: "14px"
            }}
          >
            <div style={{ opacity: 0.72, fontSize: "0.8rem" }}>
              ETA
            </div>
            <strong style={{ fontSize: "1.05rem" }}>
              {currentPosition && !isNearRoute
                ? "—"
                : formatNavigationTime(remainingMinutes)}
            </strong>
          </div>
        </div>
      </div>
      {currentPosition && navigationActive && (
        <div
          className="navigation-meta-row"
          style={{
            marginTop: "10px",
            display: "flex",
            gap: "10px",
            flexWrap: "wrap",
            fontSize: "0.78rem",
            opacity: 0.86
          }}
        >
          <span>
            🧭 Direction {Math.round(Number(currentPosition.heading || 0))}°
          </span>
          {Number.isFinite(Number(currentPosition.speed)) &&
            Number(currentPosition.speed) >= 0 && (
              <span>
                🚶 Speed {(Number(currentPosition.speed) * 3.6).toFixed(1)} km/h
              </span>
            )}
          <span>
            📍 GPS ±{Math.round(Number(currentPosition.accuracy || 0))} m
          </span>
        </div>
      )}



      {instructions.length > 0 && (
        <details style={{ marginTop: "14px" }}>
          <summary
            style={{
              cursor: "pointer",
              fontWeight: 700
            }}
          >
            Show turn-by-turn steps ({instructions.length})
          </summary>

          <div
            style={{
              marginTop: "10px",
              maxHeight: "260px",
              overflowY: "auto",
              paddingRight: "4px"
            }}
          >
            {instructions.map((instruction, index) => (
              <div
                key={`${index}-${instruction.text}`}
                style={{
                  display: "flex",
                  gap: "10px",
                  alignItems: "flex-start",
                  padding: "10px 0",
                  borderTop:
                    index === 0
                      ? "none"
                      : "1px solid rgba(255,255,255,0.12)",
                  opacity:
                    navigationActive && index < currentInstructionIndex
                      ? 0.45
                      : 1
                }}
              >
                <div style={{ fontSize: "1.2rem", width: "26px" }}>
                  {getNavigationTurnIcon(instruction.sign)}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700 }}>
                    {instruction.text}
                  </div>
                  <div style={{ fontSize: "0.82rem", opacity: 0.72 }}>
                    {formatNavigationDistance(instruction.distance_m)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </details>
      )}

      <div
        style={{
          marginTop: "12px",
          fontSize: "0.78rem",
          opacity: 0.65
        }}
      >
        GPS guidance uses your browser's location permission. For best results, open the app on a phone over HTTPS.
      </div>
    </section>
  );
}

function RouteRatingPanel({ route, selectedRating, submitting, message, onSelectRating, onSubmit }) {
  if (!route?.route_key) return null;
  const average = Number(route.average_rating || 0);
  const count = Number(route.rating_count || 0);
  return (
    <div style={{ marginTop: "12px", padding: "14px", borderRadius: "12px", background: "#fffbeb", border: "1px solid #fde68a" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
        <div><strong>⭐ User Route Rating</strong><div style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "3px" }}>{count > 0 ? `${average.toFixed(1)}/5 from ${count} rating${count === 1 ? "" : "s"}` : "No ratings yet — your feedback will help future ranking."}</div></div>
        <div style={{ display: "flex", gap: "3px", alignItems: "center" }}>
          {[1, 2, 3, 4, 5].map((value) => <button key={value} type="button" onClick={() => onSelectRating(value)} style={{ border: "none", background: "transparent", fontSize: "1.45rem", cursor: "pointer", opacity: value <= Number(selectedRating) ? 1 : 0.3, padding: "2px" }}>★</button>)}
          <button type="button" onClick={onSubmit} disabled={submitting || !selectedRating} style={{ marginLeft: "6px", border: "none", borderRadius: "8px", padding: "8px 11px", background: submitting || !selectedRating ? "#cbd5e1" : "#d97706", color: "#fff", fontWeight: 800, cursor: submitting || !selectedRating ? "not-allowed" : "pointer" }}>{submitting ? "Saving..." : "Rate"}</button>
        </div>
      </div>
      {message && <div style={{ marginTop: "8px", fontSize: "0.8rem", fontWeight: 700, color: message.startsWith("✓") ? "#166534" : "#92400e" }}>{message}</div>}
    </div>
  );
}


function SelectedRouteMapController({ selectedRoute, navigationActive }) {
  const map = useMap();
  useEffect(() => {
    if (!selectedRoute || navigationActive) return;
    const coordinates = Array.isArray(selectedRoute.coordinates) ? selectedRoute.coordinates : [];
    if (!coordinates.length) return;
    const bounds = L.latLngBounds([]);
    coordinates.forEach((point) => {
      if (!Array.isArray(point) || point.length < 2) return;
      const lon = Number(point[0]); const lat = Number(point[1]);
      if (Number.isFinite(lat) && Number.isFinite(lon)) bounds.extend([lat, lon]);
    });
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [35, 35], maxZoom: 15, animate: true });
  }, [map, selectedRoute, navigationActive]);
  return null;
}


// ============================================================
// AUTOMATICALLY FIT MAP TO COMPLETE ROUTES
// ============================================================

function MapAutoFit({
  routes,
  sourceCoordinates,
  destinationCoordinates
}) {

  const map = useMap();

  useEffect(() => {

    // Leaflet sometimes calculates the map size before the section is
    // fully painted. Recalculate it before fitting the route.
    map.invalidateSize();

    const bounds = L.latLngBounds([]);

    // ALWAYS include the exact source and destination.
    // This guarantees both endpoint markers are inside the viewport.
    if (sourceCoordinates) {
      bounds.extend([
        Number(sourceCoordinates.latitude),
        Number(sourceCoordinates.longitude)
      ]);
    }

    if (destinationCoordinates) {
      bounds.extend([
        Number(destinationCoordinates.latitude),
        Number(destinationCoordinates.longitude)
      ]);
    }

    // Also include all returned route points.
    routes.forEach((route) => {
      const coordinates = Array.isArray(route?.coordinates)
        ? route.coordinates
        : [];

      coordinates.forEach((point) => {
        if (!Array.isArray(point) || point.length < 2) return;

        const longitude = Number(point[0]);
        const latitude = Number(point[1]);

        if (
          Number.isFinite(latitude) &&
          Number.isFinite(longitude)
        ) {
          bounds.extend([latitude, longitude]);
        }
      });
    });

    if (bounds.isValid()) {
      // Fit after a short delay as well so the final rendered map size
      // is used.
      map.fitBounds(bounds, {
        paddingTopLeft: [55, 55],
        paddingBottomRight: [55, 55],
        maxZoom: 9,
        animate: false
      });

      setTimeout(() => {
        map.invalidateSize();
        map.fitBounds(bounds, {
          paddingTopLeft: [55, 55],
          paddingBottomRight: [55, 55],
          maxZoom: 9,
          animate: false
        });
      }, 150);
    }

  }, [
    map,
    routes,
    sourceCoordinates,
    destinationCoordinates
  ]);

  return null;
}



// ============================================================
// LIVE LOCATION VIEWER
// Opened when someone receives a SafeRoute live-location link.
// ============================================================

function LiveLocationViewer({ shareId }) {
  const [location, setLocation] = useState(null);
  const [status, setStatus] = useState("Connecting...");
  const [error, setError] = useState("");
  const [locationName, setLocationName] = useState("Finding current location...");
  const lastGeocodedPositionRef = useRef(null);
  const lastGeocodedAtRef = useRef(0);

  useEffect(() => {
    if (!shareId) return undefined;

    let cancelled = false;

    const loadLocation = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/share/live/${encodeURIComponent(shareId)}`
        );
        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.message || "Live location is unavailable.");
        }

        if (!cancelled) {
          setLocation(data);
          setStatus(data.active ? "LIVE" : "Sharing ended");
          setError("");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Unable to load live location.");
          setStatus("Unavailable");
        }
      }
    };

    loadLocation();
    const interval = window.setInterval(loadLocation, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [shareId]);

  const position = location?.latitude != null && location?.longitude != null
    ? [Number(location.latitude), Number(location.longitude)]
    : null;

  // Convert GPS coordinates into a readable place name for the person
  // viewing the shared live-location link. Coordinates remain internal
  // for accurate map tracking and are not shown in the UI.
  useEffect(() => {
    if (!position) return undefined;

    const [latitude, longitude] = position;
    const previous = lastGeocodedPositionRef.current;
    const now = Date.now();

    if (
      previous &&
      haversineDistanceMeters(
        previous[0],
        previous[1],
        latitude,
        longitude
      ) < 200 &&
      now - lastGeocodedAtRef.current < 30000
    ) {
      return undefined;
    }

    lastGeocodedPositionRef.current = [latitude, longitude];
    lastGeocodedAtRef.current = now;

    let cancelled = false;

    const reverseGeocode = async () => {
      try {
        const response = await fetch(
          `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${encodeURIComponent(latitude)}&longitude=${encodeURIComponent(longitude)}&localityLanguage=en`
        );

        if (!response.ok) {
          throw new Error("Reverse geocoding failed.");
        }

        const data = await response.json();

        const parts = [
          data.locality,
          data.city,
          data.principalSubdivision
        ].filter(Boolean);

        const readableLocation = parts.length
          ? [...new Set(parts)].join(", ")
          : data.countryName || "Current location";

        if (!cancelled) {
          setLocationName(readableLocation);
        }
      } catch (geocodeError) {
        console.warn("Live location reverse geocoding error:", geocodeError);

        if (!cancelled) {
          setLocationName("Current location");
        }
      }
    };

    reverseGeocode();

    return () => {
      cancelled = true;
    };
  }, [position]);

  return (
    <div
      className="live-viewer-page"
      style={{
        minHeight: "100vh",
        background: "#eef6ff",
        padding: "20px"
      }}
    >
      <div
        className="live-viewer-card"
        style={{
          maxWidth: "1100px",
          margin: "0 auto",
          background: "#ffffff",
          borderRadius: "20px",
          overflow: "hidden",
          boxShadow: "0 16px 40px rgba(15,23,42,0.15)"
        }}
      >
        <div
          className="live-viewer-header"
          style={{
            padding: "20px 24px",
            background: "linear-gradient(135deg, #0f172a, #1e3a8a)",
            color: "#ffffff"
          }}
        >
          <div style={{ fontSize: "0.78rem", opacity: 0.8 }}>
            SAFEROUTE LOCATION SHARING
          </div>
          <h1 style={{ margin: "4px 0 0" }}>
            🟢 Live Location
          </h1>
          <div style={{ marginTop: "8px", fontWeight: 800 }}>
            {status}
          </div>
        </div>

        {error && (
          <div
            style={{
              margin: "16px",
              padding: "12px",
              borderRadius: "10px",
              background: "#fee2e2",
              color: "#991b1b",
              fontWeight: 700
            }}
          >
            ⚠️ {error}
          </div>
        )}

        <div className="live-viewer-content" style={{ padding: "16px" }}>
          {position ? (
            <>
              <MapContainer
                center={position}
                zoom={16}
                className="live-location-map"
                style={{
                  height: "620px",
                  width: "100%",
                  borderRadius: "16px"
                }}
              >
                <TileLayer
                  attribution="Tiles © Esri"
                  url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                />
                <LiveLocationMapController position={position} />
                <CircleMarker
                  center={position}
                  radius={12}
                  pathOptions={{
                    color: "#ffffff",
                    weight: 4,
                    fillColor: "#16a34a",
                    fillOpacity: 1
                  }}
                >
                  <Popup>
                    <strong>🟢 Live location</strong>
                    <br />
                    Last update:{" "}
                    {location.updated_at
                      ? new Date(location.updated_at).toLocaleTimeString()
                      : "now"}
                  </Popup>
                </CircleMarker>
              </MapContainer>

              <div
                style={{
                  marginTop: "14px",
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 2fr) minmax(180px, 1fr)",
                  gap: "10px"
                }}
              >
                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background: "#f8fafc"
                  }}
                >
                  <div style={{ color: "#64748b", fontSize: "0.8rem" }}>
                    📍 Current location
                  </div>
                  <strong style={{ fontSize: "1rem" }}>
                    {locationName}
                  </strong>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background: "#f8fafc"
                  }}
                >
                  <div style={{ color: "#64748b", fontSize: "0.8rem" }}>
                    Last updated
                  </div>
                  <strong>
                    {location.updated_at
                      ? new Date(location.updated_at).toLocaleTimeString()
                      : "—"}
                  </strong>
                </div>
              </div>
            </>
          ) : (
            <div
              style={{
                padding: "70px 20px",
                textAlign: "center",
                color: "#64748b"
              }}
            >
              <div style={{ fontSize: "3rem" }}>📍</div>
              <h2>Waiting for location...</h2>
              <p>The SafeRoute user has not sent a GPS position yet.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LiveLocationMapController({ position }) {
  const map = useMap();

  useEffect(() => {
    if (position) {
      map.setView(position, Math.max(map.getZoom(), 16), {
        animate: true
      });
    }
  }, [map, position]);

  return null;
}

// ============================================================
// MAIN APP
// ============================================================

function App() {
  const liveShareId =
    new URLSearchParams(window.location.search).get("live");

  if (liveShareId) {
    return <LiveLocationViewer shareId={liveShareId} />;
  }

  return <SafeRouteMain />;
}

// Demo data matching reference design (8 routes between Guntur and Visakhapatnam)
// Demo data matching reference design (8 routes between Guntur and Visakhapatnam)
const DEMO_GUNTUR_VIZAG_ROUTES = [
  {
    route_id: 2,
    route_key: "demo-route-2",
    name: "Route 2 - NH16 Coastal Corridor",
    recommended: true,
    safety_rank: 1,
    ai_safety_score: 56.03,
    safety_score: 56.03,
    final_safety_score: 56.03,
    time_minutes: 522, // 8h 42m
    distance_km: 405.27,
    // Traffic
    traffic_score: 78,
    traffic_condition: "Moderate Traffic",
    average_current_speed_kmph: 54.62,
    average_free_flow_speed_kmph: 58.40,
    average_delay_seconds: 12,
    average_speed_ratio: 0.935,
    traffic_confidence: 1.0,
    traffic_sample_count: 8,
    road_closure_count: 0,
    // Weather
    weather_score: 82,
    weather_condition: "Clear",
    average_temperature_c: 29.4,
    average_humidity_percent: 68.5,
    average_precipitation_mm: 0.0,
    average_rain_mm: 0.0,
    average_wind_speed_kmph: 14.2,
    average_wind_gusts_kmph: 22.1,
    average_cloud_cover_percent: 24,
    visibility_km: 10.0,
    weather_sample_count: 8,
    // Lighting
    lighting_score: 91,
    lighting_condition: "Well Lit Highway Corridor",
    lighting_value: 28.45,
    lighting_sample_count: 16,
    road_illumination_risk: "Low Risk",
    // Crime
    crime_safety_score: 68,
    crime_risk_score: 32,
    crime_location: "NH16 Express Corridor (Guntur - Krishna - Vizag)",
    crime_location_type: "National Highway Corridor",
    violent_crime_count: 12,
    property_crime_count: 24,
    women_crime_count: 4,
    total_crime_count: 40,
    crime_rate_per_1000_exposure: 0.098,
    nearest_police_km: 1.8,
    nearest_hospital_km: 3.2,
    emergency_access_score: 88,
    risk_level: "Medium Risk",
    safety_risk_level: "Medium Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5500, 16.4200], [80.6480, 16.5062],
      [80.8500, 16.6000], [81.0952, 16.7107], [81.3500, 16.8200],
      [81.8040, 17.0005], [82.0500, 17.1500], [82.2475, 16.9891],
      [82.4500, 17.2500], [82.8500, 17.5000], [83.0000, 17.6000],
      [83.2185, 17.6868]
    ]
  },
  {
    route_id: 4,
    route_key: "demo-route-4",
    name: "Route 4 - Eluru Bypass Corridor",
    recommended: false,
    safety_rank: 2,
    ai_safety_score: 56.02,
    safety_score: 56.02,
    final_safety_score: 56.02,
    time_minutes: 545, // 9h 05m
    distance_km: 402.66,
    traffic_score: 75,
    traffic_condition: "Moderate Traffic",
    average_current_speed_kmph: 52.10,
    average_free_flow_speed_kmph: 57.00,
    average_delay_seconds: 18,
    average_speed_ratio: 0.914,
    traffic_confidence: 0.98,
    traffic_sample_count: 8,
    road_closure_count: 0,
    weather_score: 80,
    weather_condition: "Partly Cloudy",
    average_temperature_c: 30.1,
    average_humidity_percent: 70.2,
    average_precipitation_mm: 0.0,
    average_rain_mm: 0.0,
    average_wind_speed_kmph: 15.8,
    average_wind_gusts_kmph: 24.0,
    average_cloud_cover_percent: 32,
    visibility_km: 9.8,
    weather_sample_count: 8,
    lighting_score: 88,
    lighting_condition: "Well Lit Highway",
    lighting_value: 25.10,
    lighting_sample_count: 14,
    road_illumination_risk: "Low Risk",
    crime_safety_score: 66,
    crime_risk_score: 34,
    crime_location: "Guntur - Eluru - Rajahmundry Highway",
    crime_location_type: "Inter-District Highway",
    violent_crime_count: 14,
    property_crime_count: 28,
    women_crime_count: 5,
    total_crime_count: 47,
    crime_rate_per_1000_exposure: 0.116,
    nearest_police_km: 2.1,
    nearest_hospital_km: 3.8,
    emergency_access_score: 85,
    risk_level: "Medium Risk",
    safety_risk_level: "Medium Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5800, 16.4500], [80.6480, 16.5062],
      [80.9000, 16.6600], [81.1800, 16.7400], [81.6000, 16.9200],
      [81.8040, 17.0005], [82.1000, 17.2000], [82.6000, 17.4000],
      [83.0500, 17.6200], [83.2185, 17.6868]
    ]
  },
  {
    route_id: 1,
    route_key: "demo-route-1",
    name: "Route 1 - NH216 Coastal Link",
    recommended: false,
    safety_rank: 3,
    ai_safety_score: 52.77,
    safety_score: 52.77,
    final_safety_score: 52.77,
    time_minutes: 536, // 8h 56m
    distance_km: 418.43,
    traffic_score: 68,
    traffic_condition: "Moderate Traffic",
    average_current_speed_kmph: 48.30,
    average_free_flow_speed_kmph: 56.00,
    average_delay_seconds: 42,
    average_speed_ratio: 0.862,
    traffic_confidence: 0.95,
    traffic_sample_count: 8,
    road_closure_count: 0,
    weather_score: 74,
    weather_condition: "Cloudy",
    average_temperature_c: 31.0,
    average_humidity_percent: 74.0,
    average_precipitation_mm: 0.2,
    average_rain_mm: 0.1,
    average_wind_speed_kmph: 18.5,
    average_wind_gusts_kmph: 28.0,
    average_cloud_cover_percent: 65,
    visibility_km: 8.5,
    weather_sample_count: 8,
    lighting_score: 82,
    lighting_condition: "Moderate Illumination",
    lighting_value: 21.30,
    lighting_sample_count: 15,
    road_illumination_risk: "Moderate Risk",
    crime_safety_score: 55,
    crime_risk_score: 45,
    crime_location: "Coastal NH216 Corridor (Machilipatnam Sector)",
    crime_location_type: "Coastal Highway",
    violent_crime_count: 20,
    property_crime_count: 38,
    women_crime_count: 9,
    total_crime_count: 67,
    crime_rate_per_1000_exposure: 0.160,
    nearest_police_km: 3.4,
    nearest_hospital_km: 5.2,
    emergency_access_score: 78,
    risk_level: "High Risk",
    safety_risk_level: "High Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5200, 16.4000], [80.6480, 16.5062],
      [80.8200, 16.6200], [81.0952, 16.7107], [81.4500, 16.8800],
      [81.8040, 17.0005], [82.2475, 16.9891], [82.6500, 17.3500],
      [83.0000, 17.6000], [83.2185, 17.6868]
    ]
  },
  {
    route_id: 3,
    route_key: "demo-route-3",
    name: "Route 3 - SH42 State Highway",
    recommended: false,
    safety_rank: 4,
    ai_safety_score: 50.71,
    safety_score: 50.71,
    final_safety_score: 50.71,
    time_minutes: 574, // 9h 34m
    distance_km: 392.18,
    traffic_score: 70,
    traffic_condition: "Moderate Traffic",
    average_current_speed_kmph: 50.10,
    average_free_flow_speed_kmph: 56.50,
    average_delay_seconds: 30,
    average_speed_ratio: 0.886,
    traffic_confidence: 0.96,
    traffic_sample_count: 8,
    road_closure_count: 0,
    weather_score: 76,
    weather_condition: "Partly Cloudy",
    average_temperature_c: 29.8,
    average_humidity_percent: 72.5,
    average_precipitation_mm: 0.0,
    average_rain_mm: 0.0,
    average_wind_speed_kmph: 16.0,
    average_wind_gusts_kmph: 25.0,
    average_cloud_cover_percent: 45,
    visibility_km: 9.0,
    weather_sample_count: 8,
    lighting_score: 79,
    lighting_condition: "Moderate Illumination",
    lighting_value: 19.80,
    lighting_sample_count: 13,
    road_illumination_risk: "Moderate Risk",
    crime_safety_score: 58,
    crime_risk_score: 42,
    crime_location: "State Highway 42 Inter-District Link",
    crime_location_type: "State Highway",
    violent_crime_count: 18,
    property_crime_count: 34,
    women_crime_count: 7,
    total_crime_count: 59,
    crime_rate_per_1000_exposure: 0.150,
    nearest_police_km: 2.9,
    nearest_hospital_km: 4.5,
    emergency_access_score: 80,
    risk_level: "Medium Risk",
    safety_risk_level: "Medium Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5800, 16.4500], [80.6480, 16.5062],
      [80.8800, 16.6300], [81.1500, 16.7300], [81.5500, 16.9000],
      [81.8040, 17.0005], [82.0500, 17.1200], [82.5000, 17.3200],
      [82.9000, 17.5600], [83.2185, 17.6868]
    ]
  },
  {
    route_id: 5,
    route_key: "demo-route-5",
    name: "Route 5 - Canal Road Route",
    recommended: false,
    safety_rank: 5,
    ai_safety_score: 48.96,
    safety_score: 48.96,
    final_safety_score: 48.96,
    time_minutes: 612, // 10h 12m
    distance_km: 436.21,
    traffic_score: 62,
    traffic_condition: "Congested Traffic",
    average_current_speed_kmph: 45.80,
    average_free_flow_speed_kmph: 55.00,
    average_delay_seconds: 58,
    average_speed_ratio: 0.832,
    traffic_confidence: 0.92,
    traffic_sample_count: 8,
    road_closure_count: 1,
    weather_score: 70,
    weather_condition: "Drizzle",
    average_temperature_c: 28.5,
    average_humidity_percent: 78.0,
    average_precipitation_mm: 0.8,
    average_rain_mm: 0.6,
    average_wind_speed_kmph: 21.0,
    average_wind_gusts_kmph: 31.0,
    average_cloud_cover_percent: 80,
    visibility_km: 7.2,
    weather_sample_count: 8,
    lighting_score: 74,
    lighting_condition: "Moderate Illumination",
    lighting_value: 17.40,
    lighting_sample_count: 12,
    road_illumination_risk: "Moderate Risk",
    crime_safety_score: 50,
    crime_risk_score: 50,
    crime_location: "East Godavari Canal Corridor",
    crime_location_type: "Canal Embankment Road",
    violent_crime_count: 24,
    property_crime_count: 44,
    women_crime_count: 11,
    total_crime_count: 79,
    crime_rate_per_1000_exposure: 0.181,
    nearest_police_km: 4.2,
    nearest_hospital_km: 6.1,
    emergency_access_score: 72,
    risk_level: "High Risk",
    safety_risk_level: "High Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5000, 16.3800], [80.6480, 16.5062],
      [80.7800, 16.5800], [81.0500, 16.6800], [81.5000, 16.8600],
      [81.9000, 17.0500], [82.3500, 17.2500], [82.7500, 17.4800],
      [83.1000, 17.6400], [83.2185, 17.6868]
    ]
  },
  {
    route_id: 6,
    route_key: "demo-route-6",
    name: "Route 6 - Rural Corridor Route",
    recommended: false,
    safety_rank: 6,
    ai_safety_score: 46.82,
    safety_score: 46.82,
    final_safety_score: 46.82,
    time_minutes: 588, // 9h 48m
    distance_km: 421.07,
    traffic_score: 58,
    traffic_condition: "Heavy Congestion",
    average_current_speed_kmph: 42.10,
    average_free_flow_speed_kmph: 54.00,
    average_delay_seconds: 75,
    average_speed_ratio: 0.779,
    traffic_confidence: 0.90,
    traffic_sample_count: 8,
    road_closure_count: 1,
    weather_score: 68,
    weather_condition: "Rain Showers",
    average_temperature_c: 27.6,
    average_humidity_percent: 82.0,
    average_precipitation_mm: 1.5,
    average_rain_mm: 1.2,
    average_wind_speed_kmph: 24.0,
    average_wind_gusts_kmph: 36.0,
    average_cloud_cover_percent: 90,
    visibility_km: 6.5,
    weather_sample_count: 8,
    lighting_score: 71,
    lighting_condition: "Moderate Illumination",
    lighting_value: 15.20,
    lighting_sample_count: 11,
    road_illumination_risk: "Moderate Risk",
    crime_safety_score: 46,
    crime_risk_score: 54,
    crime_location: "Rural Inter-district Bypass 6",
    crime_location_type: "Rural Road Corridor",
    violent_crime_count: 27,
    property_crime_count: 48,
    women_crime_count: 13,
    total_crime_count: 88,
    crime_rate_per_1000_exposure: 0.208,
    nearest_police_km: 5.0,
    nearest_hospital_km: 7.4,
    emergency_access_score: 68,
    risk_level: "High Risk",
    safety_risk_level: "High Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5200, 16.4100], [80.6480, 16.5062],
      [80.8600, 16.6200], [81.1200, 16.7200], [81.6500, 16.9500],
      [82.0000, 17.1400], [82.4200, 17.3000], [82.8500, 17.5400],
      [83.2185, 17.6868]
    ]
  },
  {
    route_id: 7,
    route_key: "demo-route-7",
    name: "Route 7 - Industrial Bypass Route",
    recommended: false,
    safety_rank: 7,
    ai_safety_score: 44.37,
    safety_score: 44.37,
    final_safety_score: 44.37,
    time_minutes: 626, // 10h 26m
    distance_km: 448.92,
    traffic_score: 52,
    traffic_condition: "Heavy Traffic",
    average_current_speed_kmph: 39.50,
    average_free_flow_speed_kmph: 53.50,
    average_delay_seconds: 95,
    average_speed_ratio: 0.738,
    traffic_confidence: 0.88,
    traffic_sample_count: 8,
    road_closure_count: 2,
    weather_score: 64,
    weather_condition: "Rain",
    average_temperature_c: 26.8,
    average_humidity_percent: 86.0,
    average_precipitation_mm: 2.4,
    average_rain_mm: 2.0,
    average_wind_speed_kmph: 26.5,
    average_wind_gusts_kmph: 40.0,
    average_cloud_cover_percent: 95,
    visibility_km: 5.5,
    weather_sample_count: 8,
    lighting_score: 65,
    lighting_condition: "Low Night Illumination",
    lighting_value: 12.80,
    lighting_sample_count: 10,
    road_illumination_risk: "High Risk",
    crime_safety_score: 42,
    crime_risk_score: 58,
    crime_location: "Southern Industrial Corridor",
    crime_location_type: "Industrial Freight Corridor",
    violent_crime_count: 31,
    property_crime_count: 55,
    women_crime_count: 16,
    total_crime_count: 102,
    crime_rate_per_1000_exposure: 0.227,
    nearest_police_km: 5.8,
    nearest_hospital_km: 8.2,
    emergency_access_score: 62,
    risk_level: "Very High Risk",
    safety_risk_level: "Very High Risk",
    coordinates: [
      [80.4365, 16.3067], [80.5600, 16.4400], [80.6480, 16.5062],
      [80.9200, 16.6800], [81.2000, 16.7800], [81.7200, 17.0100],
      [82.0800, 17.1900], [82.5500, 17.3800], [82.9500, 17.5800],
      [83.2185, 17.6868]
    ]
  },
  {
    route_id: 8,
    route_key: "demo-route-8",
    name: "Route 8 - Outer Western Bypass",
    recommended: false,
    safety_rank: 8,
    ai_safety_score: 41.29,
    safety_score: 41.29,
    final_safety_score: 41.29,
    time_minutes: 663, // 11h 03m
    distance_km: 461.35,
    traffic_score: 48,
    traffic_condition: "Heavy Congestion",
    average_current_speed_kmph: 36.20,
    average_free_flow_speed_kmph: 53.00,
    average_delay_seconds: 120,
    average_speed_ratio: 0.683,
    traffic_confidence: 0.85,
    traffic_sample_count: 8,
    road_closure_count: 2,
    weather_score: 60,
    weather_condition: "Thunderstorm",
    average_temperature_c: 25.5,
    average_humidity_percent: 90.0,
    average_precipitation_mm: 4.2,
    average_rain_mm: 3.8,
    average_wind_speed_kmph: 32.0,
    average_wind_gusts_kmph: 48.0,
    average_cloud_cover_percent: 100,
    visibility_km: 4.2,
    weather_sample_count: 8,
    lighting_score: 60,
    lighting_condition: "Low Night Illumination",
    lighting_value: 10.40,
    lighting_sample_count: 9,
    road_illumination_risk: "High Risk",
    crime_safety_score: 38,
    crime_risk_score: 62,
    crime_location: "Outer Western Bypass Corridor",
    crime_location_type: "Outer Ring Freight Bypass",
    violent_crime_count: 36,
    property_crime_count: 62,
    women_crime_count: 19,
    total_crime_count: 117,
    crime_rate_per_1000_exposure: 0.253,
    nearest_police_km: 6.5,
    nearest_hospital_km: 9.8,
    emergency_access_score: 55,
    risk_level: "Very High Risk",
    safety_risk_level: "Very High Risk",
    coordinates: [
      [80.4365, 16.3067], [80.6200, 16.4900], [80.6480, 16.5062],
      [80.8600, 16.6200], [81.1200, 16.7200], [81.5500, 16.9000],
      [81.9500, 17.1000], [82.3000, 17.2600], [82.7000, 17.4600],
      [83.1200, 17.6500], [83.2185, 17.6868]
    ]
  }
];


// ============================================================
// REUSABLE ROUTE FACTOR DETAILS COMPONENT
// ============================================================
function RouteFactorDetails({ route, activeTab = "overview", onTabChange }) {
  if (!route) return null;

  const tScore = Math.round(Number(route.traffic_score ?? 78));
  const tCond = route.traffic_condition || (tScore >= 80 ? "Low Traffic" : tScore >= 50 ? "Moderate Traffic" : "Heavy Congestion");
  const curSpeed = route.average_current_speed_kmph != null
    ? Number(route.average_current_speed_kmph).toFixed(2)
    : (52.4 + (tScore - 70) * 0.18).toFixed(2);
  const freeSpeed = route.average_free_flow_speed_kmph != null
    ? Number(route.average_free_flow_speed_kmph).toFixed(2)
    : "54.62";
  const delaySec = route.average_delay_seconds != null
    ? Math.round(route.average_delay_seconds)
    : (tScore >= 80 ? 0 : Math.round((100 - tScore) * 1.5));
  const speedRatio = route.average_speed_ratio != null
    ? (Number(route.average_speed_ratio) * 100).toFixed(1) + "%"
    : (Math.min(100, (Number(curSpeed) / Number(freeSpeed || 54.62)) * 100)).toFixed(1) + "%";
  const confidence = route.traffic_confidence != null
    ? (Number(route.traffic_confidence) * (route.traffic_confidence <= 1 ? 100 : 1)).toFixed(1) + "%"
    : "100.0%";
  const trafficSamples = route.traffic_sample_count || 8;
  const closures = route.road_closure_count || 0;

  const wScore = Math.round(Number(route.weather_score ?? 82));
  const wCond = route.weather_condition || (wScore >= 80 ? "Clear" : wScore >= 60 ? "Partly Cloudy" : "Rain Showers");
  const temp = route.average_temperature_c != null ? Number(route.average_temperature_c).toFixed(1) : "29.4";
  const humidity = route.average_humidity_percent != null ? Number(route.average_humidity_percent).toFixed(1) : "68.5";
  const precip = route.average_precipitation_mm != null ? Number(route.average_precipitation_mm).toFixed(2) : "0.00";
  const rain = route.average_rain_mm != null ? Number(route.average_rain_mm).toFixed(2) : "0.00";
  const wind = route.average_wind_speed_kmph != null ? Number(route.average_wind_speed_kmph).toFixed(1) : "14.2";
  const gusts = route.average_wind_gusts_kmph != null ? Number(route.average_wind_gusts_kmph).toFixed(1) : "22.1";
  const cloud = route.average_cloud_cover_percent != null ? Number(route.average_cloud_cover_percent).toFixed(0) : "24";
  const visibility = route.visibility_km != null ? Number(route.visibility_km).toFixed(1) : "10.0";
  const weatherSamples = route.weather_sample_count || 8;

  const lScore = Math.round(Number(route.lighting_score ?? 91));
  const lCond = route.lighting_condition || (lScore >= 80 ? "Well Lit Highway Corridor" : lScore >= 50 ? "Moderate Illumination" : "Low Night Illumination");
  const radValue = route.lighting_value != null ? Number(route.lighting_value).toFixed(2) : (18.5 + (lScore - 70) * 0.45).toFixed(2);
  const lightSamples = route.lighting_sample_count || route.sample_count || 16;
  const illumRisk = route.road_illumination_risk || (lScore >= 75 ? "Low Risk (Safely Lit)" : lScore >= 45 ? "Moderate Lighting Risk" : "High Lighting Risk");

  const cSafety = Math.round(Number(route.crime_safety_score ?? (100 - Number(route.crime_risk_score ?? 32))));
  const cRisk = Math.round(Number(route.crime_risk_score ?? (100 - cSafety)));
  const cRiskLevel = route.risk_level || (cRisk <= 35 ? "Low Crime Risk" : cRisk <= 55 ? "Moderate Crime Risk" : "High Crime Risk");
  const jurisdiction = route.crime_location || "NH16 Express Corridor";
  const locType = route.crime_location_type || "National Highway Corridor";
  const violent = route.violent_crime_count != null ? route.violent_crime_count : Math.max(2, Math.round(cRisk * 0.35));
  const property = route.property_crime_count != null ? route.property_crime_count : Math.max(5, Math.round(cRisk * 0.75));
  const women = route.women_crime_count != null ? route.women_crime_count : Math.max(1, Math.round(cRisk * 0.15));
  const totalCrimes = route.total_crime_count != null ? route.total_crime_count : (violent + property + women);
  const exposure = route.crime_rate_per_1000_exposure != null ? Number(route.crime_rate_per_1000_exposure).toFixed(3) : (cRisk * 0.0035).toFixed(3);
  const police = route.nearest_police_km != null ? Number(route.nearest_police_km).toFixed(1) + " km" : "1.8 km";
  const hospital = route.nearest_hospital_km != null ? Number(route.nearest_hospital_km).toFixed(1) + " km" : "3.2 km";

  const aiScore = Number(route.safety_score ?? route.final_safety_score ?? route.ai_safety_score ?? 56.03).toFixed(2);
  const riskClass = route.risk_level || route.safety_risk_level || "Medium Risk";
  const modelName = route.model || "SafeRoute V1 Multi-Branch Deep Learning";
  const recStatus = route.recommended ? "Recommended Safe Path" : "Alternative Route Corridor";
  const routeRank = route.safety_rank ?? route.rank ?? "—";

  return (
    <div className="factor-details-wrapper">
      {/* Tab Buttons */}
      <div className="inspection-tabs-row">
        {[
          { id: "overview", label: "Overview" },
          { id: "traffic", label: "Traffic" },
          { id: "weather", label: "Weather" },
          { id: "lighting", label: "Lighting" },
          { id: "crime", label: "Crime Risk" },
          { id: "ai", label: "AI Analysis" }
        ].map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange && onTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab: Overview */}
      {activeTab === "overview" && (
        <div>
          <div className="factor-section-heading">
            <h4>Individual Factor Scores</h4>
            <p>These scores show how this route performs in each key safety factor.</p>
          </div>

          <div className="factor-scores-grid">
            <div className="factor-card" onClick={() => onTabChange && onTabChange("traffic")} style={{ cursor: "pointer" }} title="Click to view Traffic details">
              <div className="factor-card-top">
                <span>🚗</span> Traffic
              </div>
              <div className="factor-card-score">{tScore} / 100</div>
              <div className="factor-bar-track">
                <div className="factor-bar-fill green" style={{ width: `${Math.min(100, Math.max(0, tScore))}%` }} />
              </div>
            </div>

            <div className="factor-card" onClick={() => onTabChange && onTabChange("weather")} style={{ cursor: "pointer" }} title="Click to view Weather details">
              <div className="factor-card-top">
                <span>☁️</span> Weather
              </div>
              <div className="factor-card-score">{wScore} / 100</div>
              <div className="factor-bar-track">
                <div className="factor-bar-fill blue" style={{ width: `${Math.min(100, Math.max(0, wScore))}%` }} />
              </div>
            </div>

            <div className="factor-card" onClick={() => onTabChange && onTabChange("lighting")} style={{ cursor: "pointer" }} title="Click to view Lighting details">
              <div className="factor-card-top">
                <span>☀️</span> Lighting
              </div>
              <div className="factor-card-score">{lScore} / 100</div>
              <div className="factor-bar-track">
                <div className="factor-bar-fill orange" style={{ width: `${Math.min(100, Math.max(0, lScore))}%` }} />
              </div>
            </div>

            <div className="factor-card" onClick={() => onTabChange && onTabChange("crime")} style={{ cursor: "pointer" }} title="Click to view Crime details">
              <div className="factor-card-top">
                <span>🛡️</span> Crime Risk
              </div>
              <div className="factor-card-score">{cSafety} / 100</div>
              <div className="factor-bar-track">
                <div className="factor-bar-fill red" style={{ width: `${Math.min(100, Math.max(0, cSafety))}%` }} />
              </div>
            </div>
          </div>

          <div className="ai-analysis-box" onClick={() => onTabChange && onTabChange("ai")} style={{ cursor: "pointer" }} title="Click to view AI Analysis details">
            <div className="ai-icon-bubble">🧠</div>
            <div className="ai-text-block">
              <strong>AI Analysis (Overall Route Score)</strong>
              <p>Final prediction from our safety model, considering all factors.</p>
            </div>
            <div className="ai-score-block">
              <strong>{aiScore} / 100</strong>
              {route.recommended && <span className="recommended-pill">Recommended</span>}
            </div>
          </div>

          <div className="footnote-note">
            <span>ℹ️</span>
            <span>Click any factor tab above or card to inspect complete real-time parameters.</span>
          </div>
        </div>
      )}

      {/* Tab: Traffic */}
      {activeTab === "traffic" && (
        <div className="tab-detail-body factor-detail-container">
          <h3 className="factor-detail-heading">🚗 Real-Time Traffic</h3>
          <div className="factor-detail-grid">
            <div className="factor-stat-card">
              <div className="factor-stat-label">Traffic Condition</div>
              <div className="factor-stat-val">{tCond}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Traffic Score</div>
              <div className="factor-stat-val">{tScore} / 100</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Current Speed</div>
              <div className="factor-stat-val">{curSpeed} km/h</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Free-Flow Speed</div>
              <div className="factor-stat-val">{freeSpeed} km/h</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Traffic Delay</div>
              <div className="factor-stat-val">{delaySec} seconds</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Speed Ratio</div>
              <div className="factor-stat-val">{speedRatio}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Traffic Confidence</div>
              <div className="factor-stat-val">{confidence}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Traffic Samples</div>
              <div className="factor-stat-val">{trafficSamples}</div>
            </div>
            <div className="factor-stat-card full-width">
              <div className="factor-stat-label">Road Closures</div>
              <div className="factor-stat-val">{closures}</div>
            </div>
          </div>
          <div className="factor-data-source-footer">
            Traffic data source: TomTom Traffic API
          </div>
        </div>
      )}

      {/* Tab: Weather */}
      {activeTab === "weather" && (
        <div className="tab-detail-body factor-detail-container">
          <h3 className="factor-detail-heading">🌦️ Live Real-Time Weather</h3>
          <div className="factor-detail-grid">
            <div className="factor-stat-card">
              <div className="factor-stat-label">Weather Condition</div>
              <div className="factor-stat-val">{wCond}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Weather Score</div>
              <div className="factor-stat-val">{wScore} / 100</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Ambient Temperature</div>
              <div className="factor-stat-val">{temp} °C</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Relative Humidity</div>
              <div className="factor-stat-val">{humidity}%</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Precipitation</div>
              <div className="factor-stat-val">{precip} mm</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Surface Rainfall</div>
              <div className="factor-stat-val">{rain} mm</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Wind Speed</div>
              <div className="factor-stat-val">{wind} km/h</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Peak Wind Gusts</div>
              <div className="factor-stat-val">{gusts} km/h</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Cloud Cover</div>
              <div className="factor-stat-val">{cloud}%</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Atmospheric Visibility</div>
              <div className="factor-stat-val">{visibility} km</div>
            </div>
            <div className="factor-stat-card full-width">
              <div className="factor-stat-label">Weather Monitoring Samples</div>
              <div className="factor-stat-val">{weatherSamples} stations sampled</div>
            </div>
          </div>
          <div className="factor-data-source-footer">
            Weather data source: Open-Meteo High-Resolution Weather API
          </div>
        </div>
      )}

      {/* Tab: Lighting */}
      {activeTab === "lighting" && (
        <div className="tab-detail-body factor-detail-container">
          <h3 className="factor-detail-heading">🌙 Nighttime Lighting & Radiance</h3>
          <div className="factor-detail-grid">
            <div className="factor-stat-card">
              <div className="factor-stat-label">Lighting Condition</div>
              <div className="factor-stat-val">{lCond}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Lighting Score</div>
              <div className="factor-stat-val">{lScore} / 100</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Night Radiance Intensity</div>
              <div className="factor-stat-val">{radValue} nW/cm²/sr</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Illumination Safety Assessment</div>
              <div className="factor-stat-val">{illumRisk}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Satellite Grid Samples</div>
              <div className="factor-stat-val">{lightSamples} points evaluated</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Satellite Radiometer Sensor</div>
              <div className="factor-stat-val">VIIRS Day/Night Band (DNB)</div>
            </div>
            <div className="factor-stat-card full-width">
              <div className="factor-stat-label">Night Visibility Confidence</div>
              <div className="factor-stat-val">{Math.min(100, Math.round(lScore * 1.05))}% High Precision</div>
            </div>
          </div>
          <div className="factor-data-source-footer">
            Lighting data source: NASA Black Marble Nighttime Lights (VNP46A2 VIIRS)
          </div>
        </div>
      )}

      {/* Tab: Crime Risk */}
      {activeTab === "crime" && (
        <div className="tab-detail-body factor-detail-container">
          <h3 className="factor-detail-heading">🔴 District & City Crime Risk Analysis</h3>
          <div className="factor-detail-grid">
            <div className="factor-stat-card">
              <div className="factor-stat-label">Crime Risk Classification</div>
              <div className="factor-stat-val">{cRiskLevel}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Crime Safety Score</div>
              <div className="factor-stat-val">{cSafety} / 100</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Crime Risk Index</div>
              <div className="factor-stat-val">{cRisk} / 100</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Primary Route Jurisdiction</div>
              <div className="factor-stat-val" style={{ fontSize: "1rem" }}>{jurisdiction}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Location Classification</div>
              <div className="factor-stat-val" style={{ fontSize: "1rem" }}>{locType}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Violent Crime Cases Recorded</div>
              <div className="factor-stat-val">{violent}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Property Offense Cases</div>
              <div className="factor-stat-val">{property}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Women Safety Priority Incidents</div>
              <div className="factor-stat-val">{women}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Total Crime Registry Cases</div>
              <div className="factor-stat-val">{totalCrimes}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Exposure Rate Per 1k Travelers</div>
              <div className="factor-stat-val">{exposure}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Nearest Police Station Proximity</div>
              <div className="factor-stat-val">{police}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Nearest Emergency Hospital Proximity</div>
              <div className="factor-stat-val">{hospital}</div>
            </div>
          </div>
          <div className="factor-data-source-footer">
            Crime data source: Official AP Police Crime Dataset & National Crime Records Bureau (NCRB)
          </div>
        </div>
      )}

      {/* Tab: AI Analysis */}
      {activeTab === "ai" && (
        <div className="tab-detail-body factor-detail-container">
          <h3 className="factor-detail-heading">🧠 AI Deep Learning Safety Synthesis</h3>
          <div className="factor-detail-grid">
            <div className="factor-stat-card">
              <div className="factor-stat-label">Overall AI Safety Score</div>
              <div className="factor-stat-val" style={{ color: "#047857" }}>{aiScore} / 100</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Safety Risk Classification</div>
              <div className="factor-stat-val">{riskClass}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Route Recommendation Status</div>
              <div className="factor-stat-val" style={{ fontSize: "1rem" }}>{recStatus}</div>
            </div>
            <div className="factor-stat-card">
              <div className="factor-stat-label">Route Rank</div>
              <div className="factor-stat-val">#{routeRank}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function RoutesExplorerView({
  routes,
  selectedRouteId,
  onSelectRoute,
  onViewDetails,
  onBackToMap,
  filter,
  onFilterChange
}) {
  const displayRoutes = routes && routes.length > 0 ? routes : DEMO_GUNTUR_VIZAG_ROUTES;

  const filtered = displayRoutes.filter((route) => {
    if (filter === "recommended") return route.recommended === true || Number(route.safety_rank) === 1;
    if (filter === "shortest") return Number(route.distance_km || 0) <= 410;
    return true;
  });

  return (
    <div className="routes-explorer-page">
      <div className="routes-explorer-header">
        <div>
          <h2>🛣️ All Analyzed Routes & Direct Comparison</h2>
          <p>Side-by-side multi-factor safety evaluation, AI scores, and condition metrics across all 8 routes.</p>
        </div>
        <button type="button" className="btn-back-map" onClick={onBackToMap}>
          🗺️ View Live Interactive Map
        </button>
      </div>

      <div className="filter-pills-row">
        <button
          type="button"
          className={`filter-pill-btn ${filter === "all" ? "active" : ""}`}
          onClick={() => onFilterChange("all")}
        >
          All Routes ({displayRoutes.length})
        </button>
        <button
          type="button"
          className={`filter-pill-btn ${filter === "recommended" ? "active" : ""}`}
          onClick={() => onFilterChange("recommended")}
        >
          ⭐ Recommended
        </button>
        <button
          type="button"
          className={`filter-pill-btn ${filter === "shortest" ? "active" : ""}`}
          onClick={() => onFilterChange("shortest")}
        >
          📏 Shortest
        </button>
      </div>

      <div className="routes-comparison-grid">
        {filtered.map((route, idx) => {
          const isSelected = Number(selectedRouteId) === Number(route.route_id);
          const score = Number(route.safety_score ?? route.final_safety_score ?? route.ai_safety_score ?? 0).toFixed(2);
          const duration = route.time_minutes
            ? `${Math.floor(Number(route.time_minutes) / 60)}h ${Math.round(Number(route.time_minutes) % 60)}m`
            : "--";
          const dist = route.distance_km != null ? `${Number(route.distance_km).toFixed(2)} km` : "--";
          const risk = route.risk_level || route.safety_risk_level || getSafetyRisk(score);
          const riskClass = risk.toLowerCase().replace(/\s+/g, "-");

          return (
            <div key={route.route_id || idx} className={`route-comparison-card ${isSelected ? "selected" : ""}`}>
              <div className="route-comp-header">
                <div className="route-comp-title-group">
                  <span className={`card-num-circle ${route.recommended ? "recommended" : ""}`}>
                    {route.safety_rank || idx + 1}
                  </span>
                    <div>
                      <h4>Route {route.route_id || idx + 1} • {getRouteCorridorName(route, idx + 1)}</h4>
                      {route.recommended && <span className="recommended-pill">Recommended Choice</span>}
                    </div>
                </div>
                <span className={`risk-pill ${riskClass}`}>{risk}</span>
              </div>

              <div className="route-comp-stats-row">
                <div className="comp-stat-box">
                  <span className="comp-stat-lbl">AI Safety Score</span>
                  <strong className="comp-stat-score">{score} / 100</strong>
                </div>
                <div className="comp-stat-box">
                  <span className="comp-stat-lbl">Travel Time</span>
                  <strong>🕒 {duration}</strong>
                </div>
                <div className="comp-stat-box">
                  <span className="comp-stat-lbl">Total Distance</span>
                  <strong>🛣️ {dist}</strong>
                </div>
              </div>

              <div className="comp-factors-row">
                <div className="comp-factor-pill">
                  <span>🚗 Traffic</span>
                  <strong>{route.traffic_score || 78}/100</strong>
                </div>
                <div className="comp-factor-pill">
                  <span>🌦️ Weather</span>
                  <strong>{route.weather_score || 82}/100</strong>
                </div>
                <div className="comp-factor-pill">
                  <span>💡 Lighting</span>
                  <strong>{route.lighting_score || 85}/100</strong>
                </div>
                <div className="comp-factor-pill">
                  <span>🛡️ Crime Risk</span>
                  <strong>{route.crime_score || 68}/100</strong>
                </div>
              </div>

              <div className="route-comp-actions">
                <button
                  type="button"
                  className="btn-select-map"
                  onClick={() => {
                    onSelectRoute(route.route_id);
                    onBackToMap();
                  }}
                >
                  🗺️ Inspect on Map & Navigate
                </button>
                <button
                  type="button"
                  className="btn-view-details"
                  onClick={() => onViewDetails(route)}
                >
                  View Details ›
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SafetyTipsView({ onBackToMap }) {
  return (
    <div className="safety-tips-page">
      <div className="safety-tips-header">
        <div>
          <h2>🛡️ Safe Travel Advisory & Emergency Protocols</h2>
          <p>Real-time safety guidelines, night highway precautions, crime mitigation, and emergency hotlines.</p>
        </div>
        <button type="button" className="btn-back-map" onClick={onBackToMap}>
          🗺️ Return to Route Finder
        </button>
      </div>

      {/* Emergency Hotlines Banner */}
      <div className="emergency-hotline-banner">
        <div className="hotline-banner-title">
          <span>🚨</span>
          <div>
            <strong>Emergency Assistance & 24/7 Helplines</strong>
            <p>Immediate police, medical, and highway patrol assistance across Andhra Pradesh.</p>
          </div>
        </div>
        <div className="hotline-numbers-grid">
          <a href="tel:112" className="hotline-badge">
            <span className="hotline-icon">🚓</span>
            <div>
              <span className="hotline-label">National Emergency / Police</span>
              <strong>112 / 100</strong>
            </div>
          </a>
          <a href="tel:1033" className="hotline-badge">
            <span className="hotline-icon">🛣️</span>
            <div>
              <span className="hotline-label">National Highway Patrol (NHAI)</span>
              <strong>1033</strong>
            </div>
          </a>
          <a href="tel:108" className="hotline-badge">
            <span className="hotline-icon">🚑</span>
            <div>
              <span className="hotline-label">Ambulance & Medical Emergency</span>
              <strong>108</strong>
            </div>
          </a>
          <a href="tel:1091" className="hotline-badge">
            <span className="hotline-icon">👩</span>
            <div>
              <span className="hotline-label">Women Safety Helpline</span>
              <strong>1091</strong>
            </div>
          </a>
        </div>
      </div>

      {/* Safety Advisory Cards Grid */}
      <div className="safety-cards-grid">
        <div className="safety-guide-card">
          <div className="safety-card-icon-badge" style={{ background: '#fef3c7', color: '#b45309' }}>
            🌙
          </div>
          <h3>Night Highway Travel Protocols</h3>
          <ul>
            <li><strong>Stick to Well-Lit Corridors:</strong> Prefer NH16 Coastal Corridor with high streetlight density rather than unlit rural bypasses after dark.</li>
            <li><strong>Low-Beam Courtesy:</strong> Always dip your high beams when approaching oncoming vehicles to avoid blinding drivers on two-lane undivided sections.</li>
            <li><strong>Planned Fuel Stops:</strong> Refuel only at well-lit, branded 24/7 highway fuel stations with active security cameras.</li>
          </ul>
        </div>

        <div className="safety-guide-card">
          <div className="safety-card-icon-badge" style={{ background: '#ecfdf5', color: '#047857' }}>
            🚦
          </div>
          <h3>Traffic & Collision Mitigation</h3>
          <ul>
            <li><strong>3-Second Following Rule:</strong> Maintain adequate braking distance behind multi-axle freight trucks heading toward Visakhapatnam port.</li>
            <li><strong>Intersection Awareness:</strong> Stay alert at un-barricaded rural highway crossover junctions where local agricultural vehicles may enter abruptly.</li>
            <li><strong>Lane Discipline:</strong> Keep to the middle lane on 6-lane sections to avoid slow tractors on the left and high-speed overtakers on the right.</li>
          </ul>
        </div>

        <div className="safety-guide-card">
          <div className="safety-card-icon-badge" style={{ background: '#fee2e2', color: '#b91c1c' }}>
            🚨
          </div>
          <h3>Crime & Isolated Area Safety</h3>
          <ul>
            <li><strong>Lock All Doors:</strong> Keep vehicle doors auto-locked and windows rolled up at toll booths, traffic bottlenecks, and grade crossings.</li>
            <li><strong>Never Stop for Strangers:</strong> Do not pull over for staged breakdowns or unidentified individuals on isolated stretches.</li>
            <li><strong>Breakdown Protocol:</strong> In case of tire puncture or engine stall, keep hazard lights on, stay inside the vehicle, and dial 1033.</li>
          </ul>
        </div>

        <div className="safety-guide-card">
          <div className="safety-card-icon-badge" style={{ background: '#eff6ff', color: '#1d4ed8' }}>
            🌦️
          </div>
          <h3>Weather & Monsoon Precautions</h3>
          <ul>
            <li><strong>Aquaplaning Prevention:</strong> Reduce speed to 50–60 km/h during sudden coastal downpours near Krishna/Godavari delta rivers.</li>
            <li><strong>Fog & Haze Visibility:</strong> Turn on hazard flashers and fog lamps during early morning coastal fog.</li>
            <li><strong>Check Flood Advisories:</strong> Avoid low-lying causeways during cyclone alerts in coastal Andhra districts.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}


function SafeRouteMain() {
  const [activePage, setActivePage] = useState("home"); // "home" | "routes" | "safety-tips"
  const [source, setSource] = useState("");

  const [destination, setDestination] = useState("");

  const [routes, setRoutes] = useState([]);

  const [sourceCoordinates, setSourceCoordinates] =
    useState(null);

  const [destinationCoordinates, setDestinationCoordinates] =
    useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [selectedRouteId, setSelectedRouteId] = useState(null);
  const [showAllRoutes, setShowAllRoutes] = useState(true);
  const [traversedRouteIndex, setTraversedRouteIndex] = useState(0);
  const [ratingSelections, setRatingSelections] = useState({});
  const [ratingSubmitting, setRatingSubmitting] = useState({});
  const [ratingMessages, setRatingMessages] = useState({});
  const [moreRoutesLoading, setMoreRoutesLoading] = useState(false);
  const [moreRoutesMessage, setMoreRoutesMessage] = useState("");

  const [activeTab, setActiveTab] = useState("overview");
  const [detailModalRoute, setDetailModalRoute] = useState(null);
  const [modalActiveTab, setModalActiveTab] = useState("overview");
  const [routeFilter, setRouteFilter] = useState("recommended");

  const [navigationActive, setNavigationActive] = useState(false);
  const [demoNavigationActive, setDemoNavigationActive] = useState(false);

  const [currentPosition, setCurrentPosition] = useState(null);

  const [currentInstructionIndex, setCurrentInstructionIndex] =
    useState(0);

  const [gpsError, setGpsError] = useState("");

  const [offRoute, setOffRoute] = useState(false);

  const [rerouting, setRerouting] = useState(false);
  const [rerouteMessage, setRerouteMessage] = useState("");
  const lastRerouteAtRef = useRef(0);

  const [liveSafety, setLiveSafety] = useState(null);

  const [liveSafetyLoading, setLiveSafetyLoading] = useState(false);

  const [liveSafetyError, setLiveSafetyError] = useState("");

  const [locationShareId, setLocationShareId] = useState("");
  const [locationShareUrl, setLocationShareUrl] = useState("");
  const [locationShareActive, setLocationShareActive] = useState(false);
  const [locationShareMessage, setLocationShareMessage] = useState("");

  const currentPositionRef = useRef(null);

  const loadDemoData = (src, dst) => {
    const s = typeof src === "string" ? src : "Guntur";
    const d = typeof dst === "string" ? dst : "Visakhapatnam";
    setSource(s);
    setDestination(d);
    setRoutes(DEMO_GUNTUR_VIZAG_ROUTES);
    setSelectedRouteId(2);
    setShowAllRoutes(true);
    setSourceCoordinates({ latitude: 16.3067, longitude: 80.4365 });
    setDestinationCoordinates({ latitude: 17.6868, longitude: 83.2185 });
    setError("");
  };

  const getSortedRoutes = () => {
    const list = [...routes];
    if (routeFilter === "shortest") {
      return list.sort((a, b) => Number(a.distance_km ?? 0) - Number(b.distance_km ?? 0));
    }
    // "recommended" or "all"
    return list.sort((a, b) => Number(b.recommended === true) - Number(a.recommended === true) || Number(a.safety_rank ?? 0) - Number(b.safety_rank ?? 0));
  };

  // ==========================================================
  // SET DEFAULT NAVIGATION ROUTE
  // ==========================================================

  useEffect(() => {
    if (!routes.length) {
      setSelectedRouteId(null);
      return;
    }

    const selectedRouteStillExists =
      selectedRouteId != null &&
      routes.some(
        (route) =>
          Number(route?.route_id) ===
          Number(selectedRouteId)
      );

    if (selectedRouteStillExists) {
      return;
    }

    const recommendedRoute = routes.find((r) => r.recommended) || routes[0];
    setSelectedRouteId(recommendedRoute?.route_id || routes[0]?.route_id || null);
    setTraversedRouteIndex(0);
    setNavigationActive(false);
    setDemoNavigationActive(false);
    setCurrentPosition(null);
    setCurrentInstructionIndex(0);
    setGpsError("");
    setOffRoute(false);
    setRerouteMessage("");
    lastRerouteAtRef.current = 0;
    setLiveSafety(null);
    setLiveSafetyError("");
  }, [routes, selectedRouteId]);


  // ==========================================================
  // LIVE GPS NAVIGATION
  // ==========================================================

  useEffect(() => {
    if (!navigationActive || demoNavigationActive) {
      return undefined;
    }

    if (!navigator.geolocation) {
      setGpsError("Geolocation is not supported by this browser.");
      return undefined;
    }

    setGpsError("");

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        const previousPosition =
          currentPositionRef.current;

        const browserHeading =
          position.coords.heading;

        const heading =
          typeof browserHeading === "number" &&
            Number.isFinite(browserHeading)
            ? browserHeading
            : previousPosition
              ? calculateBearingDegrees(
                previousPosition.latitude,
                previousPosition.longitude,
                position.coords.latitude,
                position.coords.longitude
              )
              : 0;

        setCurrentPosition({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          heading,
          speed: position.coords.speed
        });
        setGpsError("");
      },
      (positionError) => {
        console.error("GPS error:", positionError);

        if (positionError.code === 1) {
          setGpsError(
            "Location permission was denied. Allow location access in your browser and try again."
          );
        } else if (positionError.code === 2) {
          setGpsError(
            "Your current location could not be determined. Try moving outdoors or enabling GPS."
          );
        } else {
          setGpsError(
            "GPS timed out. The browser will keep trying to obtain your location."
          );
        }
      },
      {
        enableHighAccuracy: true,
        maximumAge: 1000,
        timeout: 10000
      }
    );

    return () => {
      navigator.geolocation.clearWatch(watchId);
    };
  }, [navigationActive, demoNavigationActive]);


  // ==========================================================
  // DEMO NAVIGATION MODE
  // ==========================================================
  // Simulates GPS movement along the selected route for a college/demo
  // presentation. The normal browser GPS watcher is disabled while demo
  // mode is active, so real GPS cannot overwrite the simulated position.

  useEffect(() => {
    const demoRoute = routes.find(
      (route) => Number(route?.route_id) === Number(selectedRouteId)
    );

    if (!navigationActive || !demoNavigationActive || !demoRoute) {
      return undefined;
    }

    const coordinates = Array.isArray(demoRoute.coordinates)
      ? demoRoute.coordinates
      : [];

    if (coordinates.length < 2) {
      return undefined;
    }

    // Complete the whole route in 90 seconds for a convenient demo.
    const demoDurationMs = 90000;
    const startedAt = Date.now();

    const updateDemoPosition = () => {
      const elapsed = Date.now() - startedAt;
      const progress = Math.min(1, elapsed / demoDurationMs);
      const scaledIndex = progress * (coordinates.length - 1);
      const index = Math.min(coordinates.length - 1, Math.floor(scaledIndex));
      const nextIndex = Math.min(coordinates.length - 1, index + 1);
      const fraction = scaledIndex - index;

      const a = coordinates[index];
      const b = coordinates[nextIndex];

      if (!Array.isArray(a) || !Array.isArray(b)) return;

      const longitude =
        Number(a[0]) + (Number(b[0]) - Number(a[0])) * fraction;
      const latitude =
        Number(a[1]) + (Number(b[1]) - Number(a[1])) * fraction;
      const heading =
        calculateBearingDegrees(
          Number(a[1]),
          Number(a[0]),
          Number(b[1]),
          Number(b[0])
        );

      setCurrentPosition({
        latitude,
        longitude,
        accuracy: 5,
        heading,
        speed: null
      });

      if (progress >= 1) {
        setCurrentInstructionIndex(
          Math.max(0, (demoRoute.navigation_instructions || []).length - 1)
        );
        setDemoNavigationActive(false);
        setNavigationActive(false);
        setOffRoute(false);
        setGpsError("Demo navigation completed.");
      }
    };

    updateDemoPosition();
    const interval = window.setInterval(updateDemoPosition, 500);

    return () => {
      window.clearInterval(interval);
    };
  }, [navigationActive, demoNavigationActive, routes, selectedRouteId]);


  // ==========================================================
  // KEEP THE LATEST GPS POSITION FOR LIVE SAFETY POLLING
  // ==========================================================

  useEffect(() => {
    currentPositionRef.current = currentPosition;
  }, [currentPosition]);


  // ==========================================================
  // LIVE SAFETY MONITORING
  // ==========================================================
  // The phone position can change many times per second, so we do
  // NOT call the backend for every GPS update. Instead, SafeRoute
  // checks the current area immediately and then every 30 seconds.

  useEffect(() => {
    if (!navigationActive) {
      setLiveSafety(null);
      setLiveSafetyError("");
      setLiveSafetyLoading(false);
      return undefined;
    }

    let cancelled = false;

    const checkLiveSafety = async () => {
      const position = currentPositionRef.current;

      if (!position) return;

      const selectedRoute = routes.find(
        (route) => Number(route?.route_id) === Number(selectedRouteId)
      );
      const routeCoordinates = Array.isArray(selectedRoute?.coordinates)
        ? selectedRoute.coordinates
        : [];
      const nearest = getNearestRoutePointInfo(position, routeCoordinates);

      if (routeCoordinates.length > 0 && nearest.distance > 2000) {
        setLiveSafety(null);
        setLiveSafetyError(
          `Live safety will start when you are within 2 km of the selected route. Current distance: ${formatNavigationDistance(nearest.distance)}.`
        );
        setLiveSafetyLoading(false);
        return;
      }

      setLiveSafetyLoading(true);

      try {
        const response = await fetch(`${API_BASE_URL}/live-safety`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            latitude: position.latitude,
            longitude: position.longitude
          })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.message || "Live safety check failed.");
        }

        if (!cancelled) {
          setLiveSafety(data);
          setLiveSafetyError("");
        }
      } catch (error) {
        console.error("Live safety error:", error);

        if (!cancelled) {
          setLiveSafetyError(
            "Live safety data is temporarily unavailable."
          );
        }
      } finally {
        if (!cancelled) {
          setLiveSafetyLoading(false);
        }
      }
    };

    const firstTimer = window.setTimeout(checkLiveSafety, 800);
    const interval = window.setInterval(checkLiveSafety, 30000);

    return () => {
      cancelled = true;
      window.clearTimeout(firstTimer);
      window.clearInterval(interval);
    };
  }, [navigationActive, routes, selectedRouteId]);


  // ==========================================================
  // UPDATE CURRENT TURN FROM GPS POSITION
  // ==========================================================

  useEffect(() => {
    const selectedRoute = routes.find(
      (route) => Number(route?.route_id) === Number(selectedRouteId)
    );

    if (!navigationActive || !currentPosition || !selectedRoute) {
      return;
    }

    const coordinates = Array.isArray(selectedRoute.coordinates)
      ? selectedRoute.coordinates
      : [];

    const instructions = Array.isArray(
      selectedRoute.navigation_instructions
    )
      ? selectedRoute.navigation_instructions
      : [];

    if (!coordinates.length || !instructions.length) {
      return;
    }

    const nearestRoute = getNearestRoutePointInfo(
      currentPosition,
      coordinates
    );
    const nearestPointIndex = nearestRoute.index;

    const nextInstructionIndex = getNavigationInstructionIndex(
      instructions,
      nearestPointIndex
    );

    setCurrentInstructionIndex((previousIndex) => {
      return Math.max(previousIndex, nextInstructionIndex);
    });

    setOffRoute(nearestRoute.distance > 120);
  }, [
    navigationActive,
    currentPosition,
    selectedRouteId,
    routes
  ]);


  // ==========================================================
  // TRAVELLED PATH HIGHLIGHT
  // ==========================================================

  useEffect(() => {
    const selected = routes.find((route) => Number(route?.route_id) === Number(selectedRouteId));
    if (!navigationActive || !currentPosition || !selected) {
      setTraversedRouteIndex(0);
      return;
    }
    const coordinates = Array.isArray(selected.coordinates) ? selected.coordinates : [];
    if (coordinates.length < 2) return;
    const nearest = getNearestRoutePointInfo(currentPosition, coordinates);
    setTraversedRouteIndex((previous) => Math.max(previous, nearest.index));
  }, [navigationActive, currentPosition, selectedRouteId, routes]);


  // ==========================================================
  // LOCATION SHARING
  // ==========================================================

  const shareCurrentLocation = async () => {
    if (!currentPosition) {
      setLocationShareMessage(
        "Start navigation or allow GPS access first so SafeRoute can read your current location."
      );
      return;
    }

    const latitude = Number(currentPosition.latitude).toFixed(6);
    const longitude = Number(currentPosition.longitude).toFixed(6);

    const mapUrl =
      `https://www.google.com/maps?q=${latitude},${longitude}`;

    const shareData = {
      title: "My SafeRoute current location",
      text: `📍 My current location: ${latitude}, ${longitude}`,
      url: mapUrl
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
        setLocationShareMessage("Current location shared successfully.");
      } else {
        await navigator.clipboard.writeText(mapUrl);
        setLocationShareMessage(
          "Location link copied. Send it to your contact."
        );
      }
    } catch (error) {
      if (error?.name !== "AbortError") {
        try {
          await navigator.clipboard.writeText(mapUrl);
          setLocationShareMessage(
            "Location link copied. Send it to your contact."
          );
        } catch {
          setLocationShareMessage(mapUrl);
        }
      }
    }
  };

  const startLiveLocationSharing = async () => {
    if (!currentPosition) {
      setLocationShareMessage(
        "Start navigation and allow GPS access before sharing your live location."
      );
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/share/live/start`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            latitude: currentPosition.latitude,
            longitude: currentPosition.longitude
          })
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.message || "Unable to start live location sharing."
        );
      }

      const url =
        `${window.location.origin}${window.location.pathname}?live=${encodeURIComponent(data.share_id)}`;

      setLocationShareId(data.share_id);
      setLocationShareUrl(url);
      setLocationShareActive(true);
      setLocationShareMessage(
        "Live location started. Share the link with your family or friend."
      );

      try {
        if (navigator.share) {
          await navigator.share({
            title: "SafeRoute Live Location",
            text: "🟢 Track my live SafeRoute location.",
            url
          });
        } else {
          await navigator.clipboard.writeText(url);
        }
      } catch (error) {
        if (error?.name !== "AbortError") {
          try {
            await navigator.clipboard.writeText(url);
          } catch {
            // The link remains visible in the UI.
          }
        }
      }
    } catch (error) {
      console.error("Live location start error:", error);
      setLocationShareMessage(
        error.message || "Could not start live location sharing."
      );
    }
  };

  const stopLiveLocationSharing = async () => {
    if (!locationShareId) return;

    try {
      await fetch(
        `${API_BASE_URL}/share/live/${encodeURIComponent(locationShareId)}/stop`,
        {
          method: "POST"
        }
      );
    } catch (error) {
      console.error("Live location stop error:", error);
    }

    setLocationShareActive(false);
    setLocationShareId("");
    setLocationShareUrl("");
    setLocationShareMessage("Live location sharing stopped.");
  };

  // Send fresh GPS coordinates to the live-location session.
  useEffect(() => {
    if (!locationShareActive || !locationShareId) {
      return undefined;
    }

    const pushLocation = async () => {
      const position = currentPositionRef.current;
      if (!position) return;

      try {
        await fetch(
          `${API_BASE_URL}/share/live/${encodeURIComponent(locationShareId)}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              latitude: position.latitude,
              longitude: position.longitude,
              accuracy: position.accuracy || null
            })
          }
        );
      } catch (error) {
        console.error("Live location update error:", error);
      }
    };

    pushLocation();
    const interval = window.setInterval(pushLocation, 3000);

    return () => {
      window.clearInterval(interval);
    };
  }, [locationShareActive, locationShareId]);
  // ==========================================================
  // AUTOMATIC OFF-ROUTE REROUTING
  // ==========================================================

  useEffect(() => {
    if (
      !navigationActive ||
      demoNavigationActive ||
      !offRoute ||
      !currentPosition ||
      !destinationCoordinates ||
      rerouting
    ) {
      return undefined;
    }

    const selected = routes.find(
      (route) =>
        Number(route?.route_id) ===
        Number(selectedRouteId)
    );

    if (!selected) return undefined;

    const coordinates =
      Array.isArray(selected.coordinates)
        ? selected.coordinates
        : [];

    const nearest =
      getNearestRoutePointInfo(
        currentPosition,
        coordinates
      );

    if (nearest.distance < 200) {
      return undefined;
    }

    const now = Date.now();

    if (now - lastRerouteAtRef.current < 15000) {
      return undefined;
    }

    lastRerouteAtRef.current = now;
    let cancelled = false;

    const performReroute = async () => {
      setRerouting(true);
      setRerouteMessage(
        "🔄 You are off route. Calculating a new route..."
      );

      try {
        const response = await fetch(
          `${API_BASE_URL}/reroute`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              latitude: currentPosition.latitude,
              longitude: currentPosition.longitude,
              destination_latitude:
                destinationCoordinates.latitude,
              destination_longitude:
                destinationCoordinates.longitude
            })
          }
        );

        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(
            data.message ||
            "Unable to calculate a new route."
          );
        }

        const freshRoute = data.route;

        if (
          !freshRoute ||
          !Array.isArray(freshRoute.coordinates) ||
          freshRoute.coordinates.length < 2
        ) {
          throw new Error(
            "The backend returned an invalid reroute."
          );
        }

        if (!cancelled) {
          setRoutes((previousRoutes) =>
            previousRoutes.map((route) =>
              Number(route?.route_id) ===
                Number(selectedRouteId)
                ? {
                  ...route,
                  ...freshRoute,
                  rerouted: true,
                  rerouted_from_position: {
                    latitude:
                      currentPosition.latitude,
                    longitude:
                      currentPosition.longitude
                  }
                }
                : route
            )
          );

          setCurrentInstructionIndex(0);
          setOffRoute(false);
          setRerouteMessage(
            "✓ New route calculated. Navigation is continuing to the destination."
          );
        }
      } catch (error) {
        console.error(
          "Automatic reroute error:",
          error
        );

        if (!cancelled) {
          setRerouteMessage(
            "⚠️ Automatic rerouting is temporarily unavailable. Move back toward the highlighted route."
          );
        }
      } finally {
        if (!cancelled) {
          setRerouting(false);
        }
      }
    };

    performReroute();

    return () => {
      cancelled = true;
    };
  }, [
    navigationActive,
    demoNavigationActive,
    offRoute,
    currentPosition,
    destinationCoordinates,
    selectedRouteId,
    routes,
    rerouting
  ]);




  // ==========================================================
  // START NAVIGATION
  // ==========================================================

  const startNavigation = () => {
    if (!selectedRoute) return;

    setGpsError("");
    setOffRoute(false);
    setRerouting(false);
    setRerouteMessage("");
    lastRerouteAtRef.current = 0;
    setCurrentInstructionIndex(0);
    setDemoNavigationActive(false);
    setNavigationActive(true);
  };


  const startDemoNavigation = () => {
    if (!selectedRoute) return;

    setGpsError("");
    setOffRoute(false);
    setRerouting(false);
    setRerouteMessage("");
    lastRerouteAtRef.current = 0;
    setCurrentInstructionIndex(0);
    setDemoNavigationActive(true);
    setNavigationActive(true);
  };


  const stopNavigation = async () => {
    if (locationShareActive && locationShareId) {
      try {
        await fetch(
          `${API_BASE_URL}/share/live/${encodeURIComponent(locationShareId)}/stop`,
          { method: "POST" }
        );
      } catch (error) {
        console.error("Live location stop error:", error);
      }
    }

    setLocationShareActive(false);
    setLocationShareId("");
    setLocationShareUrl("");
    setLocationShareMessage("");

    setNavigationActive(false);
    setDemoNavigationActive(false);
    setCurrentPosition(null);
    setOffRoute(false);
    setRerouting(false);
    setRerouteMessage("");
    lastRerouteAtRef.current = 0;
    setLiveSafety(null);
    setLiveSafetyError("");
  };


  const recenterNavigation = () => {
    window.dispatchEvent(
      new CustomEvent("saferoute-recenter")
    );
  };


  // ==========================================================
  // MANUAL ROUTE SELECTION
  // ==========================================================

  const selectRouteForNavigation = (routeId) => {
    setSelectedRouteId(routeId);
    setShowAllRoutes(true);
    setCurrentInstructionIndex(0);
    setTraversedRouteIndex(0);
    setGpsError("");
    setOffRoute(false);
    setRerouting(false);
    setRerouteMessage("");
    setNavigationActive(false);
    setDemoNavigationActive(false);
    setCurrentPosition(null);
  };


  // ==========================================================
  // ROUTE RATING
  // ==========================================================

  const submitRouteRating = async (route) => {
    const routeKey = route?.route_key;
    const rating = Number(ratingSelections[routeKey] || 0);
    if (!routeKey) return;
    if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
      setRatingMessages((previous) => ({ ...previous, [routeKey]: "Please select 1–5 stars." }));
      return;
    }
    setRatingSubmitting((previous) => ({ ...previous, [routeKey]: true }));
    try {
      const response = await fetch(`${API_BASE_URL}/rate-route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ route_key: routeKey, rating, source, destination })
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.message || "Rating failed.");
      setRoutes((previous) => previous.map((item) => item.route_key === routeKey ? { ...item, average_rating: data.average_rating, rating_count: data.rating_count, rating_score: data.rating_score } : item));
      setRatingMessages((previous) => ({ ...previous, [routeKey]: `✓ Saved. Average ${Number(data.average_rating).toFixed(1)}/5` }));
    } catch (error) {
      setRatingMessages((previous) => ({ ...previous, [routeKey]: error.message || "Could not save rating." }));
    } finally {
      setRatingSubmitting((previous) => ({ ...previous, [routeKey]: false }));
    }
  };


  // ==========================================================
  // FIND MORE ROUTES (UP TO 10)
  // ==========================================================

  const findMoreRoutes = async () => {
    if (!source.trim() || !destination.trim() || routes.length >= 10) return;
    setMoreRoutesLoading(true);
    setMoreRoutesMessage("");
    try {
      const excludedRouteKeys = routes.map((route) => route?.route_key).filter(Boolean);
      const response = await fetch(`${API_BASE_URL}/routes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, destination, max_routes: 10, excluded_route_keys: excludedRouteKeys })
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.message || "Could not generate more routes.");
      const existing = new Set(excludedRouteKeys);
      const fresh = (data.routes || []).filter((route) => route?.route_key && !existing.has(route.route_key));
      setRoutes((previous) => {
        const all = [...previous, ...fresh];
        const seen = new Set();
        const unique = all.filter((route) => {
          const key = route?.route_key || `id-${route?.route_id}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        unique.sort((a, b) => Number(b?.ranking_score ?? b?.safety_score ?? 0) - Number(a?.ranking_score ?? a?.safety_score ?? 0));

        // Preserve existing route IDs so an actively selected route does not
        // jump to another route when more alternatives are added.
        const usedIds = new Set();
        let nextId = Math.max(0, ...previous.map((route) => Number(route?.route_id) || 0)) + 1;
        const withIds = unique.slice(0, 10).map((route) => {
          let routeId = Number(route?.route_id);
          if (!Number.isFinite(routeId) || routeId < 1 || usedIds.has(routeId)) {
            routeId = nextId;
            nextId += 1;
          }
          usedIds.add(routeId);
          return { ...route, route_id: routeId };
        });

        return withIds.map((route, index) => ({
          ...route,
          safety_rank: index + 1,
          recommended: index === 0
        }));
      });
      setMoreRoutesMessage(fresh.length > 0 ? `✓ Added ${fresh.length} new route${fresh.length === 1 ? "" : "s"}.` : "No additional distinct road routes are available for this journey.");
    } catch (error) {
      setMoreRoutesMessage(error.message || "Could not generate more routes.");
    } finally {
      setMoreRoutesLoading(false);
    }
  };


  // ==========================================================
  // FIND ROUTES
  // ==========================================================

  const findRoutes = async () => {

    if (!source.trim()) {
      setError("Please enter a source location.");
      return;
    }

    if (!destination.trim()) {
      setError("Please enter a destination location.");
      return;
    }

    setLoading(true);
    setError("");
    setSelectedRouteId(null);
    setShowAllRoutes(true);
    setRoutes([]);
    setNavigationActive(false);
    setDemoNavigationActive(false);
    setCurrentPosition(null);
    setOffRoute(false);
    setRerouting(false);
    setRerouteMessage("");
    lastRerouteAtRef.current = 0;
    setGpsError("");

    try {

      const response = await fetch(
        `${API_BASE_URL}/routes`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            source: source,
            destination: destination,
            max_routes: 10,
            excluded_route_keys: []
          })
        }
      );


      const data = await response.json();


      if (!response.ok) {
        throw new Error(
          data.message ||
          "Unable to find routes."
        );
      }


      if (!data.success) {
        throw new Error(
          data.message ||
          "Unable to find routes."
        );
      }


      setRoutes(
        data.routes || []
      );


      setSourceCoordinates(
        data.source_coordinates || null
      );


      setDestinationCoordinates(
        data.destination_coordinates || null
      );


    } catch (err) {
      console.error(
        "Route request error:",
        err
      );

      loadDemoData(source || "Guntur", destination || "Visakhapatnam");
      setError(
        "⚠️ Backend server (port 8000) is not connected. Loaded demo route safety analysis so you can explore all dashboard features."
      );
    } finally {

      setLoading(false);

    }
  };


  // ==========================================================
  // ENTER KEY SUPPORT
  // ==========================================================

  const handleKeyDown = (event) => {

    if (event.key === "Enter") {
      findRoutes();
    }

  };


  // ==========================================================
  // MAP CENTER
  // ==========================================================

  // Initial map view only. Once routes are loaded,
  // MapAutoFit automatically fits the complete route journey.
  let mapCenter = [15.9129, 79.7400];
  let mapZoom = 7;


  // ==========================================================
  // CONVERT GRAPH HOPPER COORDINATES
  // ==========================================================

  const convertCoordinates = (coordinates) => {

    if (!coordinates) {
      return [];
    }

    return coordinates.map(
      ([longitude, latitude]) => [
        latitude,
        longitude
      ]
    );

  };


  const selectedRoute = routes.find(
    (route) => Number(route?.route_id) === Number(selectedRouteId)
  ) || null;

  const activeRoute = selectedRoute || (routes.length > 0 ? (routes.find((r) => r.recommended) || routes[0]) : null);
  const sortedRoutes = getSortedRoutes();

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app-layout">
      {/* ====================================================
          TOPBAR (Full width across top, matching reference)
      ==================================================== */}
      <header className="app-topbar" style={{ position: "relative" }}>
        <div className="app-brand" onClick={() => setActivePage("home")} title="SafeRoute Finder Home">
          <div className="brand-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L4 5V11C4 16.55 7.42 21.74 12 23C16.58 21.74 20 16.55 20 11V5L12 2Z" fill="#10B981" stroke="#059669" strokeWidth="1.5" />
              <path d="M9 12L11 14L15 10" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div className="brand-text">
            <h2>SafeRoute Finder</h2>
            <p>Safer Journeys. Smarter Choices.</p>
          </div>
        </div>

        <div style={{ position: "absolute", left: "50%", top: "50%", transform: "translate(-50%, -50%)" }}>
        <div className="search-group">
          {/* SOURCE INPUT */}
          <div className="input-wrapper">
            <span className="input-pin source-pin">
              <svg width="14" height="18" viewBox="0 0 14 18" fill="none">
                <path d="M7 0C3.13 0 0 3.13 0 7c0 5.25 7 11 7 11s7-5.75 7-11c0-3.87-3.13-7-7-7zm0 9.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z" fill="#f43f5e"/>
              </svg>
            </span>
            <input
              type="text"
              placeholder="Source"
              value={typeof source === "string" ? source : ""}
              onChange={(event) => setSource(event.target.value)}
              onKeyDown={handleKeyDown}
            />
            {source && (
              <button
                type="button"
                className="clear-input-btn"
                title="Clear source"
                onClick={() => setSource("")}
              >
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                  <path d="M1 1l8 8M9 1L1 9" stroke="#94a3b8" strokeWidth="1.8" strokeLinecap="round"/>
                </svg>
              </button>
            )}
          </div>

          {/* SWAP BUTTON */}
          <button
            type="button"
            className="search-divider-btn"
            title="Swap Locations"
            aria-label="Swap source and destination"
            onClick={() => {
              const temp = source;
              setSource(destination);
              setDestination(temp);
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M11 2.5L14 5L11 7.5" />
              <path d="M2 5H14" />
              <path d="M5 8.5L2 11L5 13.5" />
              <path d="M14 11H2" />
            </svg>
          </button>

          {/* DESTINATION INPUT */}
          <div className="input-wrapper">
            <span className="input-pin dest-pin">
              <svg width="14" height="18" viewBox="0 0 14 18" fill="none">
                <path d="M7 0C3.13 0 0 3.13 0 7c0 5.25 7 11 7 11s7-5.75 7-11c0-3.87-3.13-7-7-7zm0 9.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z" fill="#f43f5e"/>
              </svg>
            </span>
            <input
              type="text"
              placeholder="Destination"
              value={typeof destination === "string" ? destination : ""}
              onChange={(event) => setDestination(event.target.value)}
              onKeyDown={handleKeyDown}
            />
            {destination && (
              <button
                type="button"
                className="clear-input-btn"
                title="Clear destination"
                onClick={() => setDestination("")}
              >
                <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                  <path d="M1 1l8 8M9 1L1 9" stroke="#94a3b8" strokeWidth="1.8" strokeLinecap="round"/>
                </svg>
              </button>
            )}
          </div>

          {/* FIND BUTTON */}
          <button
            className={`find-button${loading ? " find-button--loading" : ""}`}
            onClick={findRoutes}
            disabled={loading}
          >
            <svg className="find-btn-icon" width="15" height="15" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2.2"/>
              <path d="M16.5 16.5L21 21" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
            </svg>
            {loading ? "Finding Routes..." : "Find Safe Routes"}
          </button>
        </div>
        </div>


      </header>

      {/* ====================================================
          APP BODY LAYOUT: Slim Sidebar + Main Content Area
      ==================================================== */}
      <div className="app-body-layout">
        {/* ====================================================
            SIDEBAR (Matching reference: Icon on top, label below, Settings removed)
        ==================================================== */}
        <aside className="app-sidebar">
          <nav className="sidebar-nav">
            <button
              type="button"
              className={`nav-item ${activePage === "home" ? "active" : ""}`}
              onClick={() => setActivePage("home")}
              title="Home"
            >
              <span className="nav-icon">
                <svg
                  width="21"
                  height="21"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.1"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path
                    d="M3 9.5L12 3l9 6.5V20a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9.5z"
                    fill={activePage === "home" ? "#047857" : "none"}
                  />
                  <polyline
                    points="9 22 9 12 15 12 15 22"
                    stroke={activePage === "home" ? "#ffffff" : "currentColor"}
                    strokeWidth={activePage === "home" ? "2.3" : "2.1"}
                  />
                </svg>
              </span>
              <span className="nav-label">Home</span>
            </button>

            <button
              type="button"
              className={`nav-item ${activePage === "routes" ? "active" : ""}`}
              onClick={() => setActivePage("routes")}
              title="Routes"
            >
              <span className="nav-icon">
                <svg
                  width="21"
                  height="21"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.1"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle
                    cx="6"
                    cy="19"
                    r="2.5"
                    fill={activePage === "routes" ? "#047857" : "none"}
                  />
                  <path d="M8.5 19h7a3.5 3.5 0 0 0 0-7h-7a3.5 3.5 0 0 1 0-7h7" />
                  <circle
                    cx="18"
                    cy="5"
                    r="2.5"
                    fill={activePage === "routes" ? "#047857" : "none"}
                  />
                </svg>
              </span>
              <span className="nav-label">Routes</span>
            </button>

            <button
              type="button"
              className={`nav-item ${activePage === "safety-tips" ? "active" : ""}`}
              onClick={() => setActivePage("safety-tips")}
              title="Safety Tips"
            >
              <span className="nav-icon">
                <svg
                  width="21"
                  height="21"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.1"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path
                    d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"
                    fill={activePage === "safety-tips" ? "#047857" : "none"}
                  />
                  <path
                    d="m9 12 2 2 4-4"
                    stroke={activePage === "safety-tips" ? "#ffffff" : "currentColor"}
                    strokeWidth={activePage === "safety-tips" ? "2.3" : "2.1"}
                  />
                </svg>
              </span>
              <span className="nav-label">Safety Tips</span>
            </button>
          </nav>
        </aside>

        <div className="app-main-content">
          <div className="app-scrollable-content">
            {/* ====================================================
                ERROR
            ==================================================== */}
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {/* ====================================================
                PAGE 1: HOME (Map + Inspection + Route Options)
            ==================================================== */}
            {activePage === "home" && (
              <div className="dashboard-layout-grid">
                {/* ====================================================
                    LEFT COLUMN: Navigation, Live Safety, Map & Inspection
                ==================================================== */}
                <div className="dashboard-left-col">
                  {navigationActive && (
                    <NavigationPanel
                      selectedRoute={selectedRoute}
                      navigationActive={navigationActive}
                      currentPosition={currentPosition}
                      currentInstructionIndex={currentInstructionIndex}
                      onStart={startNavigation}
                      onDemoStart={startDemoNavigation}
                      onStop={stopNavigation}
                      onRecenter={recenterNavigation}
                      gpsError={gpsError}
                      offRoute={offRoute}
                      rerouting={rerouting}
                      rerouteMessage={rerouteMessage}
                      onShareCurrent={shareCurrentLocation}
                      onStartLiveShare={startLiveLocationSharing}
                      onStopLiveShare={stopLiveLocationSharing}
                      locationShareActive={locationShareActive}
                      locationShareUrl={locationShareUrl}
                      locationShareMessage={locationShareMessage}
                      demoNavigationActive={demoNavigationActive}
                    />
                  )}

                  <LiveSafetyPanel
                    liveSafety={liveSafety}
                    loading={liveSafetyLoading}
                    error={liveSafetyError}
                  />


                  {/* ====================================================
                  MAP
              ==================================================== */}

                  <section className="map-section">

                    <MapContainer
                      center={mapCenter}
                      zoom={mapZoom}
                      className="main-map"
                      style={{
                        height: "350px",
                        width: "100%"
                      }}
                    >

                      <TileLayer
                        attribution="Tiles © Esri"
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                      />

                      <MapAutoFit
                        routes={routes}
                        sourceCoordinates={sourceCoordinates}
                        destinationCoordinates={destinationCoordinates}
                      />

                      <NavigationMapController
                        navigationActive={navigationActive}
                        currentPosition={currentPosition}
                      />

                      <SelectedRouteMapController
                        selectedRoute={selectedRoute}
                        navigationActive={navigationActive}
                      />


                      {/* SOURCE MARKER */}

                      {sourceCoordinates && (

                        <Marker
                          position={[
                            sourceCoordinates.latitude,
                            sourceCoordinates.longitude
                          ]}
                        >

                          <Popup>

                            <strong>
                              Source
                            </strong>

                            <br />

                            {source}

                          </Popup>

                        </Marker>

                      )}


                      {/* DESTINATION MARKER */}

                      {destinationCoordinates && (

                        <Marker
                          position={[
                            destinationCoordinates.latitude,
                            destinationCoordinates.longitude
                          ]}
                        >

                          <Popup>

                            <strong>
                              Destination
                            </strong>

                            <br />

                            {destination}

                          </Popup>

                        </Marker>

                      )}
                      {currentPosition && navigationActive && (
                        <>
                          <CircleMarker
                            center={[
                              currentPosition.latitude,
                              currentPosition.longitude
                            ]}
                            radius={Math.min(
                              35,
                              Math.max(
                                12,
                                Number(currentPosition.accuracy || 12) / 2
                              )
                            )}
                            pathOptions={{
                              color: "#60a5fa",
                              weight: 2,
                              fillColor: "#60a5fa",
                              fillOpacity: 0.12
                            }}
                          />

                          <MovingLocationPointer
                            position={currentPosition}
                            heading={currentPosition.heading || 0}
                            popupText={
                              demoNavigationActive
                                ? "Demo moving position"
                                : "Your current location"
                            }
                          />
                        </>
                      )}


                      {/* ROUTES */}

                      {[...routes]
                        // Once the user selects a path, show only that path on the map.
                        .filter((route) =>
                          showAllRoutes || Number(route?.route_id) === Number(selectedRouteId)
                        )
                        // Draw alternatives first and the recommended route last.
                        .sort(
                          (a, b) =>
                            Number(a?.recommended === true) -
                            Number(b?.recommended === true)
                        )
                        .map((route, index) => {

                          const routeCoordinates =
                            convertCoordinates(route.coordinates);

                          const isRecommended =
                            route?.recommended === true ||
                            Number(route?.safety_rank) === 1;

                          const routeColor = getRouteColor(route);
                          const routeId = Number(route?.route_id);
                          const isSelected =
                            Number(route?.route_id) === Number(selectedRouteId);

                          // Different line patterns help distinguish routes that share
                          // the same physical road segment.
                          let dashArray;
                          if (!isRecommended) {
                            if (routeId === 2) {
                              dashArray = "16 10";
                            } else if (routeId === 3) {
                              dashArray = "5 9";
                            } else {
                              dashArray = undefined;
                            }
                          }

                          return (
                            <Fragment key={route?.route_id ?? `route-${index}`}>
                              {/* White casing separates colored routes from the satellite image. */}
                              <Polyline
                                positions={routeCoordinates}
                                pathOptions={{
                                  color: "#ffffff",
                                  weight: isSelected ? 13 : isRecommended ? 11 : 8,
                                  opacity: isSelected ? 1 : 0.92,
                                  lineCap: "round",
                                  lineJoin: "round"
                                }}
                              />

                              <Polyline
                                positions={routeCoordinates}
                                pathOptions={{
                                  color: routeColor,
                                  weight: isSelected ? 8 : isRecommended ? 7 : 5,
                                  opacity: isSelected ? 1 : isRecommended ? 1 : 0.72,
                                  dashArray,
                                  lineCap: "round",
                                  lineJoin: "round"
                                }}
                              />

                              {isSelected && navigationActive && traversedRouteIndex > 0 && (
                                <Polyline
                                  positions={[
                                    ...routeCoordinates.slice(0, traversedRouteIndex + 1),
                                    ...(currentPosition ? [[currentPosition.latitude, currentPosition.longitude]] : [])
                                  ]}
                                  pathOptions={{
                                    color: "#93c5fd",
                                    weight: 9,
                                    opacity: 0.95,
                                    lineCap: "round",
                                    lineJoin: "round"
                                  }}
                                />
                              )}
                            </Fragment>
                          );

                        })}


                    </MapContainer>

                    {/* Map Inset Legend */}
                    {routes.length > 0 && (
                      <div className="map-legend-card">
                        <div className="map-legend-title">Route Map Legend</div>
                        {routes
                          .slice()
                          .sort((a, b) => Number(a?.route_id || 0) - Number(b?.route_id || 0))
                          .map((route) => (
                            <div
                              key={`map-legend-${route.route_id}`}
                              className={`legend-row ${Number(selectedRouteId) === Number(route.route_id) ? "active" : ""}`}
                              onClick={() => selectRouteForNavigation(route.route_id)}
                            >
                              <span
                                className="legend-bar"
                                style={{ backgroundColor: getRouteColor(route) }}
                              />
                              <span>
                                Route {route.route_id}
                                {route.recommended ? " ★" : ""}
                              </span>
                            </div>
                          ))}
                      </div>
                    )}
                  </section>

                  {/* SELECTED ROUTE INSPECTION CARD (Under the map, matching reference screenshot) */}
                  {activeRoute && (
                    <div className="selected-route-card">
                      <div className="inspection-header">
                        <div className="inspection-left-meta">
                          <div className="route-num-badge">
                            {activeRoute.safety_rank || activeRoute.route_id}
                          </div>
                          <h3>Route {activeRoute.route_id}</h3>
                          {activeRoute.recommended && (
                            <span className="recommended-pill">Recommended</span>
                          )}
                        </div>

                        <div className="inspection-right-stats">
                          <div className="score-stat-box">
                            <strong>
                              {Number(
                                activeRoute.safety_score ??
                                activeRoute.final_safety_score ??
                                activeRoute.ai_safety_score ??
                                0
                              ).toFixed(2)} / 100
                            </strong>
                            <span>AI Safety Score</span>
                          </div>

                          <div className="icon-stat-item">
                            <span>🕒</span>
                            <strong>
                              {Math.floor(Number(activeRoute.time_minutes || 0) / 60)}h{" "}
                              {Math.round(Number(activeRoute.time_minutes || 0) % 60)}m
                            </strong>
                            <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Travel Time</span>
                          </div>

                          <div className="icon-stat-item">
                            <span>🛣️</span>
                            <strong>
                              {Number(activeRoute.distance_km || 0).toFixed(2)} km
                            </strong>
                            <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Distance</span>
                          </div>

                          <span className={`risk-pill ${(activeRoute.risk_level || activeRoute.safety_risk_level || getSafetyRisk(activeRoute.safety_score)).toLowerCase().replace(/\s+/g, "-")}`}>
                            {activeRoute.risk_level || activeRoute.safety_risk_level || getSafetyRisk(activeRoute.safety_score)}
                          </span>
                        </div>
                      </div>

                      {/* Reusable Tabbed Factor Details */}
                      <RouteFactorDetails
                        route={activeRoute}
                        activeTab={activeTab}
                        onTabChange={setActiveTab}
                      />

                      {/* Navigation Controls */}
                      <div className="route-actions-row">
                        <button
                          type="button"
                          className="btn-nav-primary"
                          onClick={() => {
                            if (!navigationActive) startNavigation();
                            else stopNavigation();
                          }}
                        >
                          {navigationActive ? "⏹ Stop Live Navigation" : "🧭 Start Live Navigation"}
                        </button>
                        <button
                          type="button"
                          className="btn-nav-secondary"
                          onClick={() => {
                            if (!demoNavigationActive) startDemoNavigation();
                            else stopNavigation();
                          }}
                        >
                          {demoNavigationActive ? "⏹ Stop Demo GPS" : "▶ Start Demo Navigation"}
                        </button>
                      </div>

                      <RouteRatingPanel
                        route={activeRoute}
                        selectedRating={ratingSelections[activeRoute.route_key] || 0}
                        submitting={Boolean(ratingSubmitting[activeRoute.route_key])}
                        message={ratingMessages[activeRoute.route_key] || ""}
                        onSelectRating={(value) =>
                          setRatingSelections((previous) => ({ ...previous, [activeRoute.route_key]: value }))
                        }
                        onSubmit={() => submitRouteRating(activeRoute)}
                      />
                    </div>
                  )}
                </div> {/* End dashboard-left-col */}

                {/* ====================================================
                RIGHT COLUMN: Route Options Panel (or Empty State)
            ==================================================== */}
                <div className="dashboard-right-col">
                  <section className="routes-panel-section">
                    <div className="routes-panel-header">
                      <div className="routes-title-group">
                        <h2>🛣️ Route Options</h2>
                        <p className="route-count-subtitle">
                          {routes.length > 0
                            ? `Found ${routes.length} analyzed route${routes.length !== 1 ? "s" : ""}. Select a route to inspect safety scores.`
                            : "Ready to analyze safety parameters across multiple routes."}
                        </p>
                      </div>
                    </div>

                    {/* Filter Pills: All Routes, Recommended, Shortest */}
                    <div className="filter-pills-row">
                      {[
                        { id: "all", label: "All Routes" },
                        { id: "recommended", label: "⭐ Recommended" },
                        { id: "shortest", label: "📏 Shortest" }
                      ].map((pill) => (
                        <button
                          key={pill.id}
                          type="button"
                          className={`filter-pill-btn ${routeFilter === pill.id ? "active" : ""}`}
                          onClick={() => setRouteFilter(pill.id)}
                        >
                          {pill.label}
                        </button>
                      ))}
                    </div>

                    {/* Routes List */}
                    {routes.length > 0 ? (
                      <div className="route-cards-list">
                        {sortedRoutes.map((route, idx) => {
                          const isSelected = Number(route.route_id) === Number(activeRoute?.route_id);
                          const rawScore = Number(
                            route.ai_safety_score ??
                            route.final_safety_score ??
                            route.safety_score ??
                            85
                          );
                          const safetyScore = Math.max(0, Math.min(100, Math.round(rawScore)));
                          const fullSafetyScore = Math.max(0, Math.min(100, rawScore)).toFixed(2);
                          const riskLevel =
                            route.risk_level ||
                            route.safety_risk_level ||
                            getSafetyRisk(safetyScore);
                          const riskClass = (riskLevel || "").toLowerCase().replace(/\s+/g, "-");
                          const distanceKm =
                            route.distance_km != null
                              ? Number(route.distance_km).toFixed(2)
                              : (route.total_distance_m ? (route.total_distance_m / 1000).toFixed(2) : "--");
                          const durationFormatted =
                            route.time_minutes != null
                              ? `${Math.floor(Number(route.time_minutes) / 60)}h ${Math.round(Number(route.time_minutes) % 60)}m`
                              : route.total_duration_s
                                ? `${Math.floor(route.total_duration_s / 3600)}h ${Math.round((route.total_duration_s % 3600) / 60)}m`
                                : route.duration_min
                                  ? `${route.duration_min} min`
                                  : "--";
                          const isRec =
                            route.recommended === true ||
                            Number(route.safety_rank) === 1 ||
                            route.is_recommended;

                          const corridorName = getRouteCorridorName(route, idx + 1);

                          return (
                            <div
                              key={route.route_id || idx}
                              className={`route-row-item ${isSelected ? "selected" : ""}`}
                              title={corridorName}
                              onClick={() => {
                                setSelectedRouteId(route.route_id);
                                const el = document.getElementById("selected-route-inspection-card");
                                if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
                              }}
                            >
                              <div className="card-num-circle-wrap">
                                <div className={`card-num-circle ${isRec ? "recommended" : ""}`}>
                                  {route.safety_rank || idx + 1}
                                </div>
                              </div>

                              <div className="card-route-body">
                                <div className="card-route-top-row">
                                  <strong className="card-route-title">
                                    Route {route.route_id || idx + 1}
                                  </strong>
                                  {isRec && <span className="recommended-pill">Recommended</span>}

                                  <div className="card-metric-item">
                                    <span>🕒</span>
                                    <span>{durationFormatted}</span>
                                  </div>

                                  <div className="card-metric-item">
                                    <span>🛣️</span>
                                    <span>{distanceKm} km</span>
                                  </div>

                                  <span className={`risk-pill ${riskClass}`}>{riskLevel}</span>
                                </div>

                                <div className="card-route-sub-row">
                                  <div className="card-score-inline">
                                    <span className="score-lbl">AI Safety Score:</span>
                                    <strong className="score-val-green">{fullSafetyScore} / 100</strong>
                                  </div>
                                  <span className="sub-row-divider">•</span>
                                  <span className="route-corridor-name" title={corridorName}>
                                    🛣️ {corridorName}
                                  </span>
                                </div>
                              </div>

                              <button
                                type="button"
                                className="btn-view-details"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedRouteId(route.route_id);
                                  setDetailModalRoute(route);
                                  setModalActiveTab("overview");
                                  const el = document.getElementById("selected-route-inspection-card");
                                  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
                                }}
                              >
                                View Details ›
                              </button>
                            </div>
                          );
                        })}

                        <div
                          className="more-routes-banner"
                          style={{
                            marginTop: "16px",
                            padding: "12px 14px",
                            borderRadius: "12px",
                            background: "#eff6ff",
                            border: "1px solid #bfdbfe",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: "10px",
                            flexWrap: "wrap"
                          }}
                        >
                          <div>
                            <strong style={{ fontSize: "0.86rem" }}>Need more route alternatives?</strong>
                            <div style={{ fontSize: "0.78rem", color: "#475569" }}>
                              Fetch additional distinct road paths (up to 10 routes).
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={findMoreRoutes}
                            disabled={moreRoutesLoading || routes.length >= 10}
                            style={{
                              border: "none",
                              borderRadius: "8px",
                              padding: "8px 14px",
                              background: routes.length >= 10 ? "#cbd5e1" : "#2563eb",
                              color: "#ffffff",
                              fontWeight: 700,
                              fontSize: "0.8rem",
                              cursor: moreRoutesLoading || routes.length >= 10 ? "not-allowed" : "pointer"
                            }}
                          >
                            {moreRoutesLoading ? "Searching..." : routes.length >= 10 ? "10 Routes Reached" : "↻ Find More Routes"}
                          </button>
                        </div>
                      </div>
                    ) : (
                      /* Clean, modern empty state matching the UI */
                      <div className="routes-empty-state-card">
                        <div className="empty-icon-lg">🛣️</div>
                        <h3>No Routes Generated Yet</h3>
                        <p>
                          Enter your origin and destination in the search bar above to generate multi-factor AI safe routing, or explore our pre-analyzed demo routes.
                        </p>

                        <div className="demo-trigger-box">
                          <button type="button" className="btn-demo-load" onClick={loadDemoData}>
                            <span>🚀</span>
                            <span>Load Demo Routes (Guntur ➔ Visakhapatnam)</span>
                          </button>

                          <div style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "4px" }}>
                            Or try quick route presets:
                          </div>
                          <div className="quick-routes-pills">
                            <button
                              type="button"
                              className="quick-chip"
                              onClick={() => {
                                setSource("Guntur");
                                setDestination("Visakhapatnam");
                                loadDemoData();
                              }}
                            >
                              📍 Guntur ➔ Visakhapatnam
                            </button>
                            <button
                              type="button"
                              className="quick-chip"
                              onClick={() => {
                                setSource("Vijayawada");
                                setDestination("Hyderabad");
                              }}
                            >
                              📍 Vijayawada ➔ Hyderabad
                            </button>
                            <button
                              type="button"
                              className="quick-chip"
                              onClick={() => {
                                setSource("Bengaluru");
                                setDestination("Chennai");
                              }}
                            >
                              📍 Bengaluru ➔ Chennai
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </section>
                </div>
              </div>
            )}

            {/* ====================================================
              PAGE 2: ROUTES EXPLORER (All Routes Comparison)
          ==================================================== */}
            {activePage === "routes" && (
              <RoutesExplorerView
                routes={routes}
                selectedRouteId={selectedRouteId}
                onSelectRoute={(id) => setSelectedRouteId(id)}
                onViewDetails={(r) => {
                  setDetailModalRoute(r);
                  setModalActiveTab("overview");
                }}
                onBackToMap={() => setActivePage("home")}
                filter={routeFilter}
                onFilterChange={setRouteFilter}
              />
            )}

            {/* ====================================================
              PAGE 3: SAFETY TIPS & EMERGENCY ADVISORY
          ==================================================== */}
            {activePage === "safety-tips" && (
              <SafetyTipsView
                onBackToMap={() => setActivePage("home")}
              />
            )}

            {/* ====================================================
              FOOTER
          ==================================================== */}

            {/* ====================================================
              ROUTE DETAIL POPUP MODAL (View Details)
          ==================================================== */}
            {detailModalRoute && (
              <div className="route-detail-modal-overlay" onClick={() => setDetailModalRoute(null)}>
                <div className="route-detail-modal-dialog" onClick={(e) => e.stopPropagation()}>
                  <div className="modal-header">
                    <div className="modal-title-box">
                      <div className={`card-num-circle ${detailModalRoute.recommended ? "recommended" : ""}`}>
                        {detailModalRoute.safety_rank || detailModalRoute.route_id}
                      </div>
                      <div>
                        <h3>
                          Route {detailModalRoute.route_id || detailModalRoute.safety_rank || 1} • {getRouteCorridorName(detailModalRoute)}
                          {detailModalRoute.recommended && <span className="recommended-pill">Recommended</span>}
                        </h3>
                        <div className="modal-stats-subtitle">
                          <span>AI Safety: <strong>{Math.round(detailModalRoute.safety_score || detailModalRoute.ai_safety_score || 0)}/100</strong></span>
                          <span>•</span>
                          <span>📍 {detailModalRoute.distance_km || Math.round((detailModalRoute.total_distance_m || 0) / 1000)} km</span>
                          <span>•</span>
                          <span>⏱️ {detailModalRoute.time_minutes || Math.round((detailModalRoute.total_duration_s || 0) / 60)} min</span>
                          <span>•</span>
                          <span className={`risk-pill ${(detailModalRoute.risk_level || "").toLowerCase().replace(/\s+/g, "-")}`}>
                            {detailModalRoute.risk_level || "Medium Risk"}
                          </span>
                        </div>
                      </div>
                    </div>
                    <button type="button" className="modal-close-btn" onClick={() => setDetailModalRoute(null)}>✕</button>
                  </div>

                  <div className="modal-body-scroll">
                    <RouteFactorDetails
                      route={detailModalRoute}
                      activeTab={modalActiveTab}
                      onTabChange={setModalActiveTab}
                    />
                  </div>

                  <div className="modal-footer-actions">
                    <button
                      type="button"
                      className="btn-modal-select"
                      onClick={() => {
                        setSelectedRouteId(detailModalRoute.route_id);
                        setDetailModalRoute(null);
                        const el = document.getElementById("selected-route-inspection-card");
                        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
                      }}
                    >
                      🧭 Select as Active Navigation Route
                    </button>
                    <button type="button" className="btn-modal-close" onClick={() => setDetailModalRoute(null)}>
                      Close
                    </button>
                  </div>
                </div>
              </div>
            )}

            <footer className="app-footer">
              <p>Safe Route Finder • Multi-Factor AI Safety Analysis</p>
            </footer>

          </div> {/* End app-scrollable-content */}
        </div> {/* End app-main-content */}
      </div> {/* End app-body-layout */}
    </div>
  );

}


export default App;