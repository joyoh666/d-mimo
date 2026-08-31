"""POSIX terminal input helpers that do not depend on ROS."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections.abc import Sequence


_ESCAPE_SEQUENCES = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    # Some terminals use SS3 sequences while application cursor mode is on.
    b"\x1bOA": "up",
    b"\x1bOB": "down",
    b"\x1bOC": "right",
    b"\x1bOD": "left",
}

_SINGLE_BYTE_KEYS = {
    b" ": "stop",
    b"q": "quit",
    b"Q": "quit",
    b"\x03": "quit",  # Ctrl-C if delivered as a byte by a terminal.
}


class KeyDecoder:
    """Decode terminal byte streams, including split arrow-key sequences."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[str]:
        """Return logical keys decoded from a newly received byte chunk."""
        self._buffer.extend(data)
        keys: list[str] = []

        while self._buffer:
            if self._buffer[0] == 0x1B:
                candidates = [
                    sequence
                    for sequence in _ESCAPE_SEQUENCES
                    if sequence.startswith(self._buffer)
                    or self._buffer.startswith(sequence)
                ]
                complete = next(
                    (
                        sequence
                        for sequence in candidates
                        if self._buffer.startswith(sequence)
                    ),
                    None,
                )
                if complete is not None:
                    keys.append(_ESCAPE_SEQUENCES[complete])
                    del self._buffer[: len(complete)]
                    continue
                if candidates:
                    break

                # Unknown escape sequence: discard ESC and keep parsing.
                del self._buffer[0]
                continue

            byte = bytes(self._buffer[:1])
            del self._buffer[0]
            logical_key = _SINGLE_BYTE_KEYS.get(byte)
            if logical_key is not None:
                keys.append(logical_key)

        return keys


class TerminalKeyboard:
    """Read keys without requiring Enter and restore terminal state on exit."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._original_attributes: Sequence[object] | None = None
        self._decoder = KeyDecoder()

    def open(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard_teleop must run in an interactive terminal")
        if self._original_attributes is None:
            self._original_attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def close(self) -> None:
        if self._original_attributes is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._original_attributes,
            )
            self._original_attributes = None

    def read_available(self) -> list[str]:
        """Read and decode every byte currently waiting on stdin."""
        chunks: list[bytes] = []
        while select.select([self._fd], [], [], 0.0)[0]:
            chunk = os.read(self._fd, 64)
            if not chunk:
                break
            chunks.append(chunk)
        return self._decoder.feed(b"".join(chunks)) if chunks else []

    def __enter__(self) -> "TerminalKeyboard":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
