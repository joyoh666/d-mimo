import unittest

from turtlebot4_keyboard_teleop.key_input import KeyDecoder


class KeyDecoderTest(unittest.TestCase):
    def test_decodes_arrow_keys_and_controls(self) -> None:
        decoder = KeyDecoder()

        self.assertEqual(
            decoder.feed(b"\x1b[A\x1b[B\x1b[C\x1b[D q"),
            ["up", "down", "right", "left", "stop", "quit"],
        )

    def test_decodes_split_escape_sequence(self) -> None:
        decoder = KeyDecoder()

        self.assertEqual(decoder.feed(b"\x1b"), [])
        self.assertEqual(decoder.feed(b"["), [])
        self.assertEqual(decoder.feed(b"A"), ["up"])

    def test_ignores_unmapped_printable_keys(self) -> None:
        decoder = KeyDecoder()

        self.assertEqual(decoder.feed(b"hello"), [])


if __name__ == "__main__":
    unittest.main()
