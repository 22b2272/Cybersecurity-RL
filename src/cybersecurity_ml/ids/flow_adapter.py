"""Aggregate packet summaries into a compact, observable flow feature schema.

Packet summaries are aggregated into observable flow fields, without filling
unavailable NSL-KDD fields with fabricated defaults.
"""
from dataclasses import dataclass, field
import time
from datetime import datetime
from typing import Any

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
]
SERVICE_BY_PORT = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 80: "http", 110: "pop_3", 143: "imap4", 443: "http_443",
    993: "imap4", 995: "pop_3", 3306: "mysql", 5432: "private",
}


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        result = float(value)
        if not result == result or abs(result) == float("inf"):
            raise ValueError("timestamp must be finite")
        return result
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise ValueError("timestamp must be a number or ISO-8601 string") from exc
    raise ValueError("packet event requires timestamp")


def _flags(value: Any) -> set[str]:
    raw = value if isinstance(value, list) else [value]
    aliases = {"SYN": "S", "ACK": "A", "RST": "R", "FIN": "F",
               "URG": "U", "PSH": "P", "ECE": "E", "CWR": "C"}
    normalized = set()
    for item in raw:
        for token in str(item or "").upper().replace("|", ",").split(","):
            token = token.strip()
            if token in aliases:
                normalized.add(aliases[token])
            else:
                normalized.update(token)
    return normalized


@dataclass
class _Flow:
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    first_seen: float
    last_seen: float
    packets: int = 0
    src_bytes: int = 0
    dst_bytes: int = 0
    flags: set[str] = field(default_factory=set)
    urgent: int = 0
    wrong_fragment: int = 0
    last_activity: float = field(default_factory=time.monotonic)


class PacketFlowFeatureAdapter:
    """Aggregate bidirectional five-tuples into observable flow records."""

    def __init__(self, idle_timeout: float = 5.0) -> None:
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self.idle_timeout = idle_timeout
        self.flows: dict[tuple, _Flow] = {}
        self.counter = 0

    def feed(self, packet: dict) -> list[dict]:
        required = {"timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_length"}
        missing = required.difference(packet)
        if missing:
            raise ValueError(f"Packet event missing fields: {', '.join(sorted(missing))}")
        now = _timestamp(packet["timestamp"])
        completed = self._expire(now)
        src_ip, dst_ip = str(packet["src_ip"]), str(packet["dst_ip"])
        try:
            src_port, dst_port = int(packet["src_port"]), int(packet["dst_port"])
            size = int(packet["packet_length"])
            payload = int(packet.get("payload_length", size))
            wrong_fragment = int(packet.get("wrong_fragment", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("ports and packet lengths must be integers") from exc
        if not 0 <= src_port <= 65535 or not 0 <= dst_port <= 65535:
            raise ValueError("ports must be between 0 and 65535")
        if size < 0 or payload < 0 or payload > size:
            raise ValueError("lengths must satisfy 0 <= payload_length <= packet_length")
        if wrong_fragment < 0:
            raise ValueError("wrong_fragment must be non-negative")
        protocol = str(packet["protocol"]).lower()
        if protocol not in {"tcp", "udp", "icmp"}:
            raise ValueError("protocol must be tcp, udp, or icmp")
        endpoints = sorted(((src_ip, src_port), (dst_ip, dst_port)))
        key = (protocol, *endpoints)
        flow = self.flows.get(key)
        if flow is None:
            self.counter += 1
            flow = _Flow(f"flow-{self.counter:06d}", src_ip, dst_ip, src_port, dst_port,
                         protocol, now, now)
            self.flows[key] = flow
        flow.last_seen = now
        flow.last_activity = time.monotonic()
        flow.packets += 1
        if src_ip == flow.src_ip and src_port == flow.src_port:
            flow.src_bytes += payload
        else:
            flow.dst_bytes += payload
        packet_flags = _flags(packet.get("tcp_flags"))
        flow.flags.update(packet_flags)
        flow.urgent += int("URG" in packet_flags or "U" in packet_flags)
        flow.wrong_fragment += wrong_fragment
        return completed

    def _to_features(self, flow: _Flow) -> dict:
        protocol_name = {"tcp": "tcp", "udp": "udp", "icmp": "icmp"}.get(flow.protocol, "other")
        if protocol_name == "icmp":
            service = "eco_i"
        else:
            # The first event may be a server response. Resolve a known service
            # on either endpoint instead of assuming it is always the dst port.
            service = (SERVICE_BY_PORT.get(flow.dst_port)
                       or SERVICE_BY_PORT.get(flow.src_port) or "private")
        if "R" in flow.flags:
            flag = "REJ"
        elif "S" in flow.flags and "A" not in flow.flags:
            flag = "S0"
        elif "A" in flow.flags and "S" in flow.flags:
            flag = "SF"
        else:
            flag = "SF"
        values = {
            "duration": max(0.0, flow.last_seen - flow.first_seen),
            "protocol_type": protocol_name,
            "service": service,
            "flag": flag,
            "src_bytes": flow.src_bytes,
            "dst_bytes": flow.dst_bytes,
            "land": int(flow.src_ip == flow.dst_ip and flow.src_port == flow.dst_port),
            "wrong_fragment": flow.wrong_fragment,
            "urgent": flow.urgent,
        }
        return {"event_id": flow.flow_id, **values}

    def _expire(self, now: float) -> list[dict]:
        expired = [key for key, flow in self.flows.items() if now - flow.last_seen >= self.idle_timeout]
        return [self._to_features(self.flows.pop(key)) for key in expired]

    def flush(self) -> list[dict]:
        """Finish the input stream and emit every remaining flow."""
        flows = [self._to_features(flow) for flow in self.flows.values()]
        self.flows.clear()
        return flows

    def expire_idle(self) -> list[dict]:
        """Emit flows whose packet stream has been idle for the configured time."""
        now = time.monotonic()
        expired = [key for key, flow in self.flows.items()
                   if now - flow.last_activity >= self.idle_timeout]
        return [self._to_features(self.flows.pop(key)) for key in expired]


# Backwards-compatible name retained for earlier project scripts.
KDDFlowFeatureAdapter = PacketFlowFeatureAdapter
