"""Shared constants for the smart-agriculture IDS experiments."""

from __future__ import annotations

BENIGN_LABEL = "Benign"
UNKNOWN_LABEL = "Unknown Attack"

FARM_FLOW_KNOWN_CLASSES = [
    BENIGN_LABEL,
    "HTTP Flood",
    "ICMP Flood",
    "MQTT Flood",
    "TCP Flood",
    "UDP Flood",
]

DEFAULT_ZERO_DAY_CLASSES = [
    "Arp Spoofing",
    "Port Scanning",
]

OPTIONAL_ZERO_DAY_CLASSES = [
    "BotNet DDoS",
]

FARM_FLOW_ALL_CLASSES = [
    BENIGN_LABEL,
    "Arp Spoofing",
    "BotNet DDoS",
    "HTTP Flood",
    "ICMP Flood",
    "MQTT Flood",
    "Port Scanning",
    "TCP Flood",
    "UDP Flood",
]

ALLOWED_CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "conn_state",
]

DEFAULT_COLUMNS_TO_DROP = [
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "history",
    "tunnel_parents",
    "local_orig",
    "local_resp",
]

FEATURE_PROFILES = {
    "low_leakage": {
        "description": "Flow statistics plus low-cardinality protocol fields.",
        "columns_to_drop": tuple(DEFAULT_COLUMNS_TO_DROP),
        "categorical_columns": tuple(ALLOWED_CATEGORICAL_COLUMNS),
        "leakage_risk": "low",
    },
    "protocol_context": {
        "description": "Low-leakage features plus Zeek connection history.",
        "columns_to_drop": (
            "id.orig_h",
            "id.orig_p",
            "id.resp_h",
            "id.resp_p",
            "tunnel_parents",
            "local_orig",
            "local_resp",
        ),
        "categorical_columns": tuple(ALLOWED_CATEGORICAL_COLUMNS + ["history"]),
        "leakage_risk": "low_to_moderate",
    },
    "endpoint_context": {
        "description": "Protocol context plus endpoint addresses and numeric ports.",
        "columns_to_drop": (
            "tunnel_parents",
            "local_orig",
            "local_resp",
        ),
        "categorical_columns": tuple(
            ALLOWED_CATEGORICAL_COLUMNS + ["history", "id.orig_h", "id.resp_h"]
        ),
        "leakage_risk": "high",
    },
}

LABEL_ALIASES = {
    "normal": BENIGN_LABEL,
    "benign": BENIGN_LABEL,
    "http_flood": "HTTP Flood",
    "icmp_flood": "ICMP Flood",
    "mqtt_flood": "MQTT Flood",
    "tcp_flood": "TCP Flood",
    "udp_flood": "UDP Flood",
    "arp_spoofing": "Arp Spoofing",
    "port_scanning": "Port Scanning",
    "botnet_ddos": "BotNet DDoS",
    "botnet_ddos_attack": "BotNet DDoS",
}
