"""Non-blocking POSIX terminal input helpers."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


_ARROW_SEQUENCES = {
    b'\x1b[A': 'up',
    b'\x1b[B': 'down',
    b'\x1b[C': 'right',
    b'\x1b[D': 'left',
    # Some terminals use SS3 escape sequences in application cursor mode.
    b'\x1bOA': 'up',
    b'\x1bOB': 'down',
    b'\x1bOC': 'right',
    b'\x1bOD': 'left',
}

TerminalAttributes = list[int | list[int | bytes]]


_SINGLE_BYTE_KEYS = {
    b' ': 'stop',
    b'q': 'quit',
    b'Q': 'quit',
    b'\x03': 'quit',  # Ctrl-C when the terminal delivers it as input.
}


class KeyDecoder:
    """Decode terminal byte streams, including split arrow sequences."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[str]:
        """Return logical key names decoded from a new byte chunk."""
        self._buffer.extend(data)
        keys: list[str] = []

        while self._buffer:
            if self._buffer[0] == 0x1B:
                possible_sequences = [
                    sequence
                    for sequence in _ARROW_SEQUENCES
                    if sequence.startswith(self._buffer)
                    or self._buffer.startswith(sequence)
                ]
                complete_sequence = next(
                    (
                        sequence
                        for sequence in possible_sequences
                        if self._buffer.startswith(sequence)
                    ),
                    None,
                )

                if complete_sequence is not None:
                    keys.append(_ARROW_SEQUENCES[complete_sequence])
                    del self._buffer[: len(complete_sequence)]
                    continue

                if possible_sequences:
                    # Wait for the rest of a split escape sequence.
                    break

                # Discard an unknown escape byte and continue parsing.
                del self._buffer[0]
                continue

            byte = bytes(self._buffer[:1])
            del self._buffer[0]
            key = _SINGLE_BYTE_KEYS.get(byte)
            if key is not None:
                keys.append(key)

        return keys


class TerminalKeyboard:
    """Read keys immediately and restore terminal state on exit."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._original_attributes: TerminalAttributes | None = None
        self._decoder = KeyDecoder()

    def open(self) -> None:  # noqa: A003
        """Put the interactive terminal into cbreak mode."""
        if not sys.stdin.isatty():
            raise RuntimeError(
                'keyboard_control must run in an interactive terminal'
            )
        if self._original_attributes is None:
            self._original_attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def close(self) -> None:
        """Restore the terminal settings captured by open()."""
        if self._original_attributes is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._original_attributes,
            )
            self._original_attributes = None

    def read_available(self) -> list[str]:
        """Read and decode all bytes currently waiting on standard input."""
        chunks: list[bytes] = []
        while select.select([self._fd], [], [], 0.0)[0]:
            chunk = os.read(self._fd, 64)
            if not chunk:
                break
            chunks.append(chunk)
        return self._decoder.feed(b''.join(chunks)) if chunks else []

    def __enter__(self) -> 'TerminalKeyboard':
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
