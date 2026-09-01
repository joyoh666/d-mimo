import gc
import time

import uhd


DEVICE_ARGS = "addr=192.168.1.2"
CLAIM_RETRY_TIMEOUT = 15.0
CLAIM_RETRY_INTERVAL = 2.0


def _is_claim_conflict(error):
    message = str(error).lower()
    return "rpc call to `claim'" in message and "claim this device again" in message


def connect_usrp(
    device_args=DEVICE_ARGS,
    timeout=CLAIM_RETRY_TIMEOUT,
    retry_interval=CLAIM_RETRY_INTERVAL,
):
    """Connect after a short-lived claim from a previous UHD session expires."""
    deadline = time.monotonic() + timeout

    while True:
        try:
            return uhd.usrp.MultiUSRP(device_args)
        except RuntimeError as error:
            if not _is_claim_conflict(error):
                raise

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit(
                    "USRP(192.168.1.2)를 다른 UHD 세션이 사용 중입니다. "
                    "실행 중인 GNU Radio/Python/UHD 프로그램을 종료한 뒤 다시 실행하세요."
                ) from None

            wait_time = min(retry_interval, remaining)
            print(
                "이전 UHD 세션의 USRP 점유가 해제되기를 기다립니다 "
                f"({wait_time:.1f}초)..."
            )
            time.sleep(wait_time)


def main():
    usrp = connect_usrp()
    try:
        usrp.set_rx_freq(2.2e9)
        usrp.set_rx_rate(1e6)
        usrp.set_rx_gain(20)

        print("RX freq:", usrp.get_rx_freq())
        print("RX rate:", usrp.get_rx_rate())
        print("RX gain:", usrp.get_rx_gain())
    finally:
        # MultiUSRP is released as soon as its final Python reference disappears.
        # Explicit collection prevents a following run from inheriting a stale claim.
        del usrp
        gc.collect()


if __name__ == "__main__":
    main()
