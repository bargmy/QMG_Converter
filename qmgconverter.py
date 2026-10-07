#!/usr/bin/env python3



from __future__ import annotations



import argparse

from dataclasses import dataclass

from pathlib import Path

from typing import List, Tuple



import numpy as np

from PIL import Image







QDIR: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, -1), (-1, -1))







ORI_DELTA = (

    0xffe0,0x0020,0xffff,0x0001,0xf800,0x0800,0xf7df,0xffdf,

    0xf7e0,0x0021,0xf7bf,0x0821,0x0820,0x0841,0xf7ff,0x0801,

    0xffc0,0x07e0,0xf820,0xf801,0x07ff,0xf7c0,0xffe1,0x0040,

    0x001f,0xffbf,0xf79f,0x0840,0xffff,0x0041,0x0861,0xef7e,

    0x1082,0xef9e,0xf81f,0xf7a0,0x0001,0xe73d,0xef9f,0x1062,

    0x0860,0x1061,0x18c3,0x07df,0xf7be,0xef5e,0xefbf,0xf79e,

    0xf7e1,0xf821,0x10a2,0xef7f,0x0842,0x07e1,0xdefc,0x1041,

    0x081f,0x0862,0x0002,0x1081,0x2104,0xf81f,0xefc0,0x07e1,

    0xfffe,0xffbe,0xe75d,0x2945,0xffa0,0xf780,0x1040,0xd6bb,

    0x0042,0x18a3,0x18e3,0x0880,0x0822,0xff9f,0x0022,0xffc1,

    0x003f,0xffde,0xe71d,0x3186,0xef5f,0xf7de,0xce7a,0x0060,

    0x10a1,0xe75e,0xf77e,0xdedc,0x39c7,0xf77f,0xc639,0x18a2,

    0xefe0,0x2124,0x1080,0x20e4,0x0061,0x1020,0x1000,0xefa0,

    0xff9e,0x07c0,0xdf1c,0xbdf8,0x1060,0xf000,0x4208,0xd69b,

    0x0881,0x5acb,0x2965,0x0882,0xef80,0xf840,0xe77f,0xa535,

    0xe73e,0xce5a,0x21ae,0xf760,0x1881,0x4a49,0xefbe,0xb5b7,

    0x0062,0xf7c1,0x083f,0x2925,0xe73f,0x4a4a,0xef3f,0x08a0,

    0xe75f,0x1042,0xad76,0x18c2,0x3126,0xf7fe,0x528a,0xc619,

    0x31a6,0xd6db,0xff7e,0x10c1,0xe77e,0xceda,0x18c1,0xb5b6,

    0x07bf,0xf83f,0x630c,0xde52,0x39e7,0xefdf,0xef5d,0x1861,

    0x10a0,0xe79f,0x9cf4,0xce9a,0xff7f,0x18a1,0x1021,0x0802,

    0xef7d,0x1882,0x3166,0x0003,0xf841,0xf7bd,0xe71f,0xdf7f,

    0x10a3,0x07c1,0x18e1,0x94b3,0x2081,0xbdd8,0x6b4d,0xef60,

    0x1083,0xdf1d,0xff7d,0x0863,0xf79d,0xdf3e,0xf77d,0x4228,

    0xad56,0xf7dd,0x20e3,0x52aa,0xb597,0x3967,0xc699,0x20c2,

    0x8c72,0xff80,0x001e,0x39a7,0xe6ff,0xff5f,0xc659,0x0082,

    0x1901,0x1063,0x4a69,0xef9d,0x0843,0xd6dc,0xfffd,0x738e,

    0x07fe,0x8431,0xffe2,0xe7a0,0xef40,0x7bf0,0xe71c,0xff9d,

    0xef3d,0xefff,0x2924,0xa515,0xd73e,0x8410,0x28c2,0x7bcf,

    0x1043,0xf75e,0x41e8,0x10c3,0xbe18,0x5aeb,0xd6dd,0xdf5f,

    0x9cd4,0x0883,0x20a2,0xf802,0x632c,0xe780,0x0083,0xdf5e,

    0xefbd,0xff5d,0x0823,0x1880,0x1022,0xdefd,0x18e4,0xefde,

)





class QMGError(RuntimeError):

    pass





class BitReader:



    def __init__(self, data: bytes):

        self.data = data

        self.bitpos = 0



    def read(self, n: int) -> int:

        value = 0

        for _ in range(n):

            byte_index = self.bitpos >> 3

            if byte_index >= len(self.data):

                raise QMGError("Unexpected end of QMG bitstream")

            bit_index = 7 - (self.bitpos & 7)

            value = (value << 1) | ((self.data[byte_index] >> bit_index) & 1)

            self.bitpos += 1

        return value





class ByteReader:

    def __init__(self, data: bytes):

        self.data = data

        self.pos = 0



    def u16le(self) -> int:

        if self.pos + 2 > len(self.data):

            raise QMGError("Unexpected end of QMG byte stream")

        value = int.from_bytes(self.data[self.pos:self.pos + 2], "little")

        self.pos += 2

        return value





@dataclass(frozen=True)

class Header:

    qversion: int

    raw_type: int

    qp: int

    mode: bool

    width: int

    height: int

    is_dynamic_table: bool

    use_extra_exception: bool

    total_frames: int

    frame_number: int

    packet_size: int

    header_size: int





def parse_header(data: bytes, offset: int = 0) -> Header:

    if offset + 24 > len(data):

        raise QMGError("Truncated QMG header")

    p = data[offset:offset + 24]

    if p[:2] != b"QM":

        raise QMGError(f"Bad QMG magic at offset 0x{offset:x}")



    qversion = p[2]

    raw_type = p[3]

    flags4 = p[4]

    flags5 = p[5]

    width = int.from_bytes(p[6:8], "little")

    height = int.from_bytes(p[8:10], "little")

    mode = bool(flags4 & 0x80)

    qp = flags4 & 0x1F

    is_dynamic_table = bool(flags5 & 0x10) if qversion > 0x0B else False

    use_extra_exception = bool(flags5 & 0x80)



    if raw_type != 0:

        raise QMGError(

            f"Unsupported QMG raw_type={raw_type}. This script currently supports RGB565 (raw_type=0) only."

        )

    if not mode:

        raise QMGError("This script currently supports animated QMG files only")

    if qversion <= 0x0B:

        raise QMGError(

            f"Unsupported older QMG qversion=0x{qversion:02x}; this build supports qversion > 0x0b"

        )

    if is_dynamic_table:

        raise QMGError("Dynamic QMG delta tables are not supported by this script")

    if use_extra_exception:

        raise QMGError("QMG use_extra_exception mode is not supported")



    alpha_position = int.from_bytes(p[12:14], "little")

    total_frames = int.from_bytes(p[16:18], "little")

    frame_number = int.from_bytes(p[18:20], "little")





    if frame_number <= 1:

        alpha_position *= 4



    if alpha_position <= 24:

        raise QMGError(f"Invalid QMG packet size: {alpha_position}")



    return Header(

        qversion=qversion,

        raw_type=raw_type,

        qp=qp,

        mode=mode,

        width=width,

        height=height,

        is_dynamic_table=is_dynamic_table,

        use_extra_exception=use_extra_exception,

        total_frames=total_frames,

        frame_number=frame_number,

        packet_size=alpha_position,

        header_size=24,

    )





def split_packets(data: bytes) -> List[bytes]:

    packets: List[bytes] = []

    pos = 0

    expected_frame = 1



    while pos < len(data):

        h = parse_header(data, pos)

        if h.frame_number != expected_frame:

            raise QMGError(

                f"Unexpected frame number {h.frame_number}; expected {expected_frame} at 0x{pos:x}"

            )

        end = pos + h.packet_size

        if end > len(data):

            raise QMGError(

                f"Frame {h.frame_number} extends beyond end of file ({h.packet_size} bytes)"

            )

        packets.append(data[pos:end])

        pos = end



        if h.frame_number >= h.total_frames:

            break

        expected_frame += 1



    if not packets:

        raise QMGError("No QMG frames found")

    return packets





def get_pixel(frame: np.ndarray, x: int, y: int) -> int:

    h, w = frame.shape

    if 0 <= x < w and 0 <= y < h:

        return int(frame[y, x])

    return 0





def decode_pixel_inter(

    copy: bool,

    bits_values: BitReader,

    bits_lengths: BitReader,

    literals: ByteReader,

    frame: np.ndarray,

    ref_x: int,

    ref_y: int,

) -> int:

    if copy:

        return get_pixel(frame, ref_x, ref_y)



    nbits = bits_lengths.read(3)

    if nbits == 7:

        return literals.u16le()



    idx = bits_values.read(nbits + 1)

    table_index = idx + (2 << nbits) - 2

    delta = ORI_DELTA[table_index]

    return (get_pixel(frame, ref_x, ref_y) + delta) & 0xFFFF





def decode_first_frame(packet: bytes) -> np.ndarray:

    h = parse_header(packet)

    hs = h.header_size



    if len(packet) < hs + 8:

        raise QMGError("First frame packet is too short")



    lengths_start = int.from_bytes(packet[hs:hs + 4], "little")

    literals_start = int.from_bytes(packet[hs + 4:hs + 8], "little")

    if not (hs + 8 <= lengths_start <= len(packet)):

        raise QMGError("Invalid first-frame secondary bitstream offset")

    if not (hs + 8 <= literals_start <= len(packet)):

        raise QMGError("Invalid first-frame literal stream offset")



    bits_values = BitReader(packet[hs + 8:])

    bits_lengths = BitReader(packet[lengths_start:])

    literals = ByteReader(packet[literals_start:])



    frame = np.zeros((h.height, h.width), dtype=np.uint16)



    for y in range(0, h.height, 4):

        for x in range(0, h.width, 4):

            mode = bits_values.read(2)

            if mode < 3:

                cbp = literals.u16le()

                k = 0

                dx, dy = QDIR[mode]

                for j in range(4):

                    for i in range(4):

                        px = x + i

                        py = y + j

                        if px < h.width and py < h.height:

                            frame[py, px] = decode_pixel_inter(

                                bool(cbp & (1 << k)),

                                bits_values,

                                bits_lengths,

                                literals,

                                frame,

                                px + dx,

                                py + dy,

                            )

                            k += 1

            elif x > 0:

                block_w = min(4, h.width - x)

                block_h = min(4, h.height - y)

                frame[y:y + block_h, x:x + block_w] = frame[y:y + block_h, x - 1:x]



    return frame





def decode_pixel(

    bits: BitReader,

    literals: ByteReader,

    ref: np.ndarray,

    ref_x: int,

    ref_y: int,

) -> int:

    if bits.read(1):

        return get_pixel(ref, ref_x, ref_y)



    nbits = bits.read(3)

    if nbits == 7:

        return literals.u16le()



    idx = bits.read(nbits + 1)

    table_index = idx + (2 << nbits) - 2

    return (get_pixel(ref, ref_x, ref_y) + ORI_DELTA[table_index]) & 0xFFFF





def copy_block(

    dst: np.ndarray,

    x: int,

    y: int,

    ref: np.ndarray,

    ref_x: int,

    ref_y: int,

    width: int,

    height: int,

) -> None:

    h, w = dst.shape

    if ref_x < 0 or ref_y < 0 or ref_x + width > w or ref_y + height > h:

        raise QMGError("QMG motion vector points outside the frame")

    dst[y:y + height, x:x + width] = ref[ref_y:ref_y + height, ref_x:ref_x + width]





def decode_block3(

    header: Header,

    bits: BitReader,

    literals: ByteReader,

    x: int,

    y: int,

    dst: np.ndarray,

    ref: np.ndarray,

    mv_x: int,

    mv_y: int,

) -> None:

    mode = bits.read(3)

    if header.qp != 0 and not bits.read(1):

        raise QMGError("Unsupported non-zero QP block")



    if mode < 3:

        dx, dy = QDIR[mode]

        for j in range(4):

            for i in range(4):

                dst[y + j, x + i] = decode_pixel(bits, literals, dst, x + i + dx, y + j + dy)

    elif mode == 3:

        if x > 0:

            dst[y:y + 4, x:x + 4] = dst[y:y + 4, x - 1:x]

    elif mode == 4:

        for j in range(4):

            for i in range(4):

                dst[y + j, x + i] = decode_pixel(bits, literals, ref, x + i, y + j)

    elif mode == 5:

        copy_block(dst, x, y, ref, x, y, 4, 4)

    elif mode == 6:

        for j in range(4):

            for i in range(4):

                dst[y + j, x + i] = decode_pixel(

                    bits, literals, ref, x + i + mv_x, y + j + mv_y

                )

    else:

        copy_block(dst, x, y, ref, x + mv_x, y + mv_y, 4, 4)





def decode_block2(

    header: Header,

    bits: BitReader,

    literals: ByteReader,

    x: int,

    y: int,

    dst: np.ndarray,

) -> None:

    mode = bits.read(2)

    if header.qp != 0 and not bits.read(1):

        raise QMGError("Unsupported non-zero QP block")



    if mode < 3:

        dx, dy = QDIR[mode]

        for j in range(4):

            for i in range(4):

                dst[y + j, x + i] = decode_pixel(bits, literals, dst, x + i + dx, y + j + dy)

    elif x > 0:

        dst[y:y + 4, x:x + 4] = dst[y:y + 4, x - 1:x]





def decode_macroblock(

    header: Header,

    bits: BitReader,

    literals: ByteReader,

    x: int,

    y: int,

    dst: np.ndarray,

    ref: np.ndarray,

) -> None:

    if bits.read(1):

        if bits.read(1):

            copy_block(dst, x, y, ref, x, y, 16, 16)

            return



        if not bits.read(1):

            mv_x = bits.read(8) - 0x7F

            mv_y = bits.read(7) - 0x3F

            if bits.read(1):

                copy_block(dst, x, y, ref, x + mv_x, y + mv_y, 16, 16)

                return

        else:

            mv_x = mv_y = 0



        for j in range(0, 16, 4):

            for i in range(0, 16, 4):

                decode_block3(header, bits, literals, x + i, y + j, dst, ref, mv_x, mv_y)

    else:

        for j in range(0, 16, 4):

            for i in range(0, 16, 4):

                decode_block2(header, bits, literals, x + i, y + j, dst)





def decode_edge_macroblock(

    header: Header,

    bits: BitReader,

    literals: ByteReader,

    x0: int,

    y0: int,

    dst: np.ndarray,

) -> None:





    if bits.read(1):

        raise QMGError("Unsupported skipped edge macroblock")



    for y in range(y0, min(y0 + 16, header.height), 4):

        for x in range(x0, min(x0 + 16, header.width), 4):

            if x + 4 <= header.width and y + 4 <= header.height:

                mode = bits.read(2)

                if mode < 3:

                    dx, dy = QDIR[mode]

                    for j in range(4):

                        for i in range(4):

                            dst[y + j, x + i] = decode_pixel(

                                bits, literals, dst, x + i + dx, y + j + dy

                            )

                elif x > 0:

                    dst[y:y + 4, x:x + 4] = dst[y:y + 4, x - 1:x]

            else:

                for j in range(4):

                    for i in range(4):

                        if x + i < header.width and y + j < header.height:

                            dst[y + j, x + i] = literals.u16le()





def decode_delta_frame(packet: bytes, previous: np.ndarray) -> np.ndarray:

    h = parse_header(packet)

    hs = h.header_size

    if previous.shape != (h.height, h.width):

        raise QMGError("Frame dimensions changed inside the QMG animation")



    if len(packet) < hs + 8:

        raise QMGError("Delta frame packet is too short")



    literals_start = int.from_bytes(packet[hs:hs + 4], "little")

    if not (hs + 8 <= literals_start <= len(packet)):

        raise QMGError("Invalid delta-frame literal stream offset")



    bits = BitReader(packet[hs + 8:])

    literals = ByteReader(packet[literals_start:])

    frame = np.zeros_like(previous)



    for y in range(0, h.height, 16):

        for x in range(0, h.width, 16):

            if h.width - x >= 16 and h.height - y >= 16:

                decode_macroblock(h, bits, literals, x, y, frame, previous)

            else:

                decode_edge_macroblock(h, bits, literals, x, y, frame)



    return frame





def rgb565_to_rgb888(frame: np.ndarray) -> np.ndarray:

    v = frame.astype(np.uint32)

    r5 = (v >> 11) & 0x1F

    g6 = (v >> 5) & 0x3F

    b5 = v & 0x1F





    r = ((r5 * 527 + 23) >> 6).astype(np.uint8)

    g = ((g6 * 259 + 33) >> 6).astype(np.uint8)

    b = ((b5 * 527 + 23) >> 6).astype(np.uint8)

    return np.dstack((r, g, b))





def save_png(frame: np.ndarray, path: Path) -> None:

    rgb = rgb565_to_rgb888(frame)

    Image.fromarray(rgb, mode="RGB").save(path, "PNG", optimize=True)







class BitWriter:



    def __init__(self) -> None:

        self._data = bytearray()

        self._current = 0

        self._bits_in_current = 0



    def write(self, value: int, nbits: int) -> None:

        for bit_index in range(nbits - 1, -1, -1):

            self._current = (self._current << 1) | ((value >> bit_index) & 1)

            self._bits_in_current += 1

            if self._bits_in_current == 8:

                self._data.append(self._current)

                self._current = 0

                self._bits_in_current = 0



    def finish(self) -> bytes:

        if self._bits_in_current:

            self._data.append(self._current << (8 - self._bits_in_current))

            self._current = 0

            self._bits_in_current = 0

        return bytes(self._data)





def _build_delta_map() -> dict[int, tuple[int, int]]:

    result: dict[int, tuple[int, int]] = {}

    for nbits in range(7):

        start = (2 << nbits) - 2

        count = 1 << (nbits + 1)

        for idx in range(count):

            value = ORI_DELTA[start + idx]

            result.setdefault(value, (nbits, idx))

    return result





DELTA_ENCODE = _build_delta_map()





def png_to_rgb565(image: Image.Image) -> np.ndarray:

    if "A" in image.getbands():

        rgba = image.convert("RGBA")

        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))

        image = Image.alpha_composite(black, rgba).convert("RGB")

    else:

        image = image.convert("RGB")



    a = np.asarray(image, dtype=np.uint16)

    r = (a[:, :, 0] >> 3) & 0x1F

    g = (a[:, :, 1] >> 2) & 0x3F

    b = (a[:, :, 2] >> 3) & 0x1F

    return ((r << 11) | (g << 5) | b).astype(np.uint16)





def _source_ref(frame: np.ndarray, x: int, y: int, mode: int) -> tuple[int, bool]:

    dx, dy = QDIR[mode]

    rx, ry = x + dx, y + dy

    valid = 0 <= rx < frame.shape[1] and 0 <= ry < frame.shape[0]

    return (int(frame[ry, rx]) if valid else 0), valid





def encode_keyframe_payload(frame: np.ndarray) -> bytes:

    height, width = frame.shape

    if width % 4 or height % 4:

        raise QMGError("PNG -> QMG currently requires width and height divisible by 4")



    values = BitWriter()

    lengths = BitWriter()

    literals = bytearray()



    for y in range(0, height, 4):

        for x in range(0, width, 4):

            if x > 0:
                edge_ok = True
                for j in range(4):
                    left = int(frame[y + j, x - 1])
                    for i in range(4):
                        if int(frame[y + j, x + i]) != left:
                            edge_ok = False
                            break
                    if not edge_ok:
                        break
                if edge_ok:
                    values.write(3, 2)
                    continue

            best_cost: int | None = None

            best_mode = 0

            for mode in range(3):

                cost = 0

                for j in range(4):

                    for i in range(4):

                        pixel = int(frame[y + j, x + i])

                        ref, valid_ref = _source_ref(frame, x + i, y + j, mode)

                        if valid_ref and pixel == ref:

                            continue

                        if not valid_ref:

                            cost += 19

                            continue

                        delta = (pixel - ref) & 0xFFFF

                        encoded = DELTA_ENCODE.get(delta)

                        cost += 3 + (encoded[0] + 1 if encoded else 16)

                if best_cost is None or cost < best_cost:

                    best_cost = cost

                    best_mode = mode



            values.write(best_mode, 2)

            cbp = 0

            block_literals: list[int] = []

            k = 0



            for j in range(4):

                for i in range(4):

                    pixel = int(frame[y + j, x + i])

                    ref, valid_ref = _source_ref(frame, x + i, y + j, best_mode)

                    if valid_ref and pixel == ref:

                        cbp |= 1 << k

                    elif not valid_ref:

                        lengths.write(7, 3)

                        block_literals.append(pixel)

                    else:

                        delta = (pixel - ref) & 0xFFFF

                        encoded = DELTA_ENCODE.get(delta)

                        if encoded is not None:

                            nbits, idx = encoded

                            lengths.write(nbits, 3)

                            values.write(idx, nbits + 1)

                        else:

                            lengths.write(7, 3)

                            block_literals.append(pixel)

                    k += 1



            literals += cbp.to_bytes(2, "little")

            for pixel in block_literals:

                literals += pixel.to_bytes(2, "little")



    values_bytes = values.finish()

    lengths_bytes = lengths.finish()

    values_end = 32 + len(values_bytes)

    lengths_start = (values_end + 3) & ~3

    lengths_end = lengths_start + len(lengths_bytes)

    literals_start = (lengths_end + 3) & ~3



    payload = bytearray(8)

    payload[0:4] = lengths_start.to_bytes(4, "little")

    payload[4:8] = literals_start.to_bytes(4, "little")

    payload += values_bytes

    payload += b"\x00" * (lengths_start - values_end)

    payload += lengths_bytes

    payload += b"\x00" * (literals_start - lengths_end)

    payload += literals

    return bytes(payload)





def encode_copy_delta_payload(width: int, height: int) -> bytes:

    if width % 16 or height % 16:

        raise QMGError(

            "Multi-frame PNG -> QMG currently requires width and height divisible by 16"

        )



    bits = BitWriter()

    macroblocks = (width // 16) * (height // 16)

    for _ in range(macroblocks):

        bits.write(0b11, 2)



    bitstream = bits.finish()

    packet_size = 32 + len(bitstream)

    payload = bytearray(8)

    payload[0:4] = packet_size.to_bytes(4, "little")

    payload[4:8] = b"\x00\x00\x00\x00"

    payload += bitstream

    while (24 + len(payload)) & 3:

        payload += b"\x00"

    payload[0:4] = (24 + len(payload)).to_bytes(4, "little")

    return bytes(payload)





def make_qmg_header(

    width: int,

    height: int,

    total_frames: int,

    frame_number: int,

    packet_size: int,

) -> bytes:

    header = bytearray(24)

    header[0:2] = b"QM"

    header[2] = 0x0F

    header[3] = 0

    header[4] = 0x80

    header[5] = 0

    header[6:8] = width.to_bytes(2, "little")

    header[8:10] = height.to_bytes(2, "little")

    header[10] = 0

    header[11] = 0x20



    if frame_number == 1:

        if packet_size % 4:

            raise QMGError("Internal encoder error: first QMG packet is not 4-byte aligned")

        stored_size = packet_size // 4

        if stored_size > 0xFFFF:

            raise QMGError(

                "Encoded first frame is too large for qversion 0x0f. "

                "Try a simpler image or lower resolution."

            )

    else:

        stored_size = packet_size

        if stored_size > 0xFFFF:

            raise QMGError("Encoded delta packet is too large")



    header[12:14] = stored_size.to_bytes(2, "little")

    header[14] = 0

    header[15] = 0

    header[16:18] = total_frames.to_bytes(2, "little")

    header[18:20] = frame_number.to_bytes(2, "little")

    header[20:22] = (0).to_bytes(2, "little")

    header[22] = 0

    header[23] = 0

    return bytes(header)





def encode_png_to_qmg(

    input_path: Path,

    output_path: Path,

    frames: int = 10,

    size: tuple[int, int] | None = None,

) -> None:

    if not 1 <= frames <= 0xFFFF:

        raise QMGError("--frames must be between 1 and 65535")



    with Image.open(input_path) as image:

        if size is not None and image.size != size:

            image = image.resize(size, Image.Resampling.LANCZOS)

        rgb565 = png_to_rgb565(image)



    height, width = rgb565.shape

    if width > 0xFFFF or height > 0xFFFF:

        raise QMGError("QMG dimensions must fit in 16-bit width/height fields")

    if width % 4 or height % 4:

        raise QMGError("PNG -> QMG currently requires dimensions divisible by 4")

    if frames > 1 and (width % 16 or height % 16):

        raise QMGError(

            "For more than one frame, PNG -> QMG currently requires dimensions divisible by 16"

        )



    key_payload = encode_keyframe_payload(rgb565)

    first_size = 24 + len(key_payload)

    padding = (-first_size) % 4

    if padding:

        key_payload += b"\x00" * padding

        first_size += padding



    delta_payload = encode_copy_delta_payload(width, height) if frames > 1 else b""



    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as out:

        out.write(make_qmg_header(width, height, frames, 1, first_size))

        out.write(key_payload)

        for frame_number in range(2, frames + 1):

            packet_size = 24 + len(delta_payload)

            out.write(make_qmg_header(width, height, frames, frame_number, packet_size))

            out.write(delta_payload)





def qmg_to_png_frames(input_path: Path, output_dir: Path) -> int:

    data = input_path.read_bytes()

    packets = split_packets(data)

    first_header = parse_header(packets[0])



    output_dir.mkdir(parents=True, exist_ok=True)



    frame = decode_first_frame(packets[0])

    save_png(frame, output_dir / "frame_0001.png")

    print(

        f"frame 1/{first_header.total_frames}: "

        f"{first_header.width}x{first_header.height} -> frame_0001.png"

    )



    for index, packet in enumerate(packets[1:], start=2):

        frame = decode_delta_frame(packet, frame)

        name = f"frame_{index:04d}.png"

        save_png(frame, output_dir / name)

        print(f"frame {index}/{first_header.total_frames}: -> {name}")



    return len(packets)





def parse_size(value: str) -> tuple[int, int]:

    try:

        width_s, height_s = value.lower().split("x", 1)

        width, height = int(width_s), int(height_s)

    except (ValueError, AttributeError) as exc:

        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT, e.g. 800x1280") from exc

    if width <= 0 or height <= 0:

        raise argparse.ArgumentTypeError("width and height must be positive")

    return width, height





def print_help() -> None:

    print("python qmg_converter.py input.qmg -o frames")

    print("python qmg_converter.py input.png -o output.qmg --frames 10 --size 800x1280")





def main() -> int:

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("input", type=Path, nargs="?")

    parser.add_argument("-o", "--output", type=Path)

    parser.add_argument("--frames", type=int, default=10)

    parser.add_argument("--size", type=parse_size)

    parser.add_argument("-h", "--help", action="store_true")

    args = parser.parse_args()



    if args.help:

        print_help()

        return 0

    if args.input is None:

        print_help()

        return 1



    suffix = args.input.suffix.lower()

    try:

        if suffix == ".qmg":

            output = args.output or Path("qmg_frames")

            count = qmg_to_png_frames(args.input, output)

            print(f"Done: extracted {count} PNG frame(s) to {output}")

        elif suffix == ".png":

            output = args.output or args.input.with_suffix(".qmg")

            encode_png_to_qmg(args.input, output, frames=args.frames, size=args.size)

            with Image.open(args.input) as image:

                target_size = args.size or image.size

            print(f"Done: encoded {args.frames}-frame {target_size[0]}x{target_size[1]} QMG -> {output}")

        else:

            raise QMGError("Input must be a .qmg or .png file")

    except (OSError, QMGError) as exc:

        parser.exit(1, f"error: {exc}\n")



    return 0





if __name__ == "__main__":

    raise SystemExit(main())

