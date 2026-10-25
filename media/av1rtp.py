"""AV1 RTP payload format (AOMedia "RTP Payload Format For AV1", v1.0): payloader + depacketizer.

Input to the payloader is one temporal unit (TU) in low-overhead bitstream format (Section 5 OBUs with
size fields), as produced by libsvtav1/libaom. Per the spec, temporal-delimiter and tile-list OBUs are
dropped and `obu_has_size_field` is cleared; each OBU element in the payload is LEB128-length-prefixed
(W=0). The depacketizer reverses this and returns a decodable TU (TD + OBUs with size fields).
"""

from __future__ import annotations

from dataclasses import dataclass

OBU_SEQUENCE_HEADER = 1
OBU_TEMPORAL_DELIMITER = 2
OBU_TILE_LIST = 8


def leb128_encode(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def leb128_decode(buf: bytes, off: int) -> tuple[int, int]:
    val = 0
    for i in range(8):
        b = buf[off + i]
        val |= (b & 0x7F) << (7 * i)
        if not b & 0x80:
            return val, off + i + 1
    raise ValueError("leb128 too long")


@dataclass
class Obu:
    obu_type: int
    header: bytes  # header byte(s) with has_size_field cleared
    payload: bytes

    def element(self) -> bytes:
        """OBU as carried in RTP: header + payload, no size field."""
        return self.header + self.payload

    def with_size(self) -> bytes:
        h = bytearray(self.header)
        h[0] |= 0x02
        return bytes(h) + leb128_encode(len(self.payload)) + self.payload


def parse_obus(tu: bytes) -> list[Obu]:
    obus = []
    off = 0
    while off < len(tu):
        h0 = tu[off]
        if h0 & 0x80:
            raise ValueError("forbidden bit set")
        otype = (h0 >> 3) & 0xF
        ext = bool(h0 & 0x04)
        has_size = bool(h0 & 0x02)
        hlen = 2 if ext else 1
        header = bytearray(tu[off : off + hlen])
        header[0] &= ~0x02 & 0xFF
        p = off + hlen
        if has_size:
            size, p = leb128_decode(tu, p)
        else:
            size = len(tu) - p
        obus.append(Obu(otype, bytes(header), tu[p : p + size]))
        off = p + size
    return obus


def parse_element(elem: bytes) -> Obu:
    h0 = elem[0]
    hlen = 2 if h0 & 0x04 else 1
    if h0 & 0x02:  # tolerate senders that keep the size field
        size, p = leb128_decode(elem, hlen)
        return Obu((h0 >> 3) & 0xF, bytes([h0 & ~0x02 & 0xFF]) + elem[1:hlen], elem[p : p + size])
    return Obu((h0 >> 3) & 0xF, elem[:hlen], elem[hlen:])


def packetize(tu: bytes, mtu_payload: int = 1100) -> list[tuple[bytes, bool]]:
    """Split one TU into RTP payloads. Returns [(payload, marker)], marker on the last packet."""
    obus = [o for o in parse_obus(tu) if o.obu_type not in (OBU_TEMPORAL_DELIMITER, OBU_TILE_LIST)]
    new_cvs = any(o.obu_type == OBU_SEQUENCE_HEADER for o in obus)
    packets: list[bytes] = []
    cur: list[bytes] = []
    cur_len = 1
    z = False
    first = True

    def flush(y: bool) -> None:
        nonlocal cur, cur_len, z, first
        agg = (0x80 if z else 0) | (0x40 if y else 0) | (0x08 if (new_cvs and first) else 0)
        packets.append(bytes([agg]) + b"".join(cur))
        cur, cur_len, first = [], 1, False

    for o in obus:
        data = o.element()
        while data:
            room = mtu_payload - cur_len
            if room < 4:  # no room for a length field + data
                flush(False)
                z = False
                room = mtu_payload - cur_len
            need = len(leb128_encode(len(data))) + len(data)
            if need <= room:
                cur.append(leb128_encode(len(data)) + data)
                cur_len += need
                data = b""
            else:
                take = room - len(leb128_encode(room))
                frag, data = data[:take], data[take:]
                cur.append(leb128_encode(len(frag)) + frag)
                cur_len += len(leb128_encode(len(frag))) + len(frag)
                flush(True)
                z = True
    if cur:
        flush(False)
    return [(p, i == len(packets) - 1) for i, p in enumerate(packets)]


class Depacketizer:
    """Reassemble TUs from in-order RTP payloads of one TU (caller groups by RTP timestamp)."""

    @staticmethod
    def assemble(payloads: list[bytes]) -> bytes:
        """Payloads must be the complete, in-order packet list for one TU. Raises ValueError if inconsistent."""
        elements: list[bytes] = []
        pending: bytes | None = None
        for i, pl in enumerate(payloads):
            agg = pl[0]
            z, y, w = bool(agg & 0x80), bool(agg & 0x40), (agg >> 4) & 0x3
            if w != 0:
                raise ValueError("only W=0 supported")
            if z != (pending is not None):
                raise ValueError(f"fragment continuity broken at packet {i}")
            off = 1
            elems = []
            while off < len(pl):
                n, off = leb128_decode(pl, off)
                elems.append(pl[off : off + n])
                off += n
            if pending is not None and elems:
                elems[0] = pending + elems[0]
                pending = None
            if y:
                pending = elems.pop()
            elements.extend(elems)
        if pending is not None:
            raise ValueError("TU ends inside a fragment")
        td = bytes([OBU_TEMPORAL_DELIMITER << 3 | 0x02, 0])
        return td + b"".join(parse_element(e).with_size() for e in elements)
