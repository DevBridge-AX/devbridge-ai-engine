"""가상 결제 재시도 정책 모듈 (라이브 검증용 fixture, 실제 프로젝트 코드 아님)."""

from __future__ import annotations

import time

MAX_RETRY_COUNT = 3
RETRY_BACKOFF_SECONDS = 2


class PaymentRetryError(Exception):
    """결제 재시도가 모두 실패했을 때 발생하는 예외."""


def retry_payment(payment_id: str, attempt: int = 0) -> bool:
    """결제 요청을 최대 MAX_RETRY_COUNT회까지 재시도한다.

    TODO: 재시도 중 발생하는 중복 결제 위험을 막기 위한 idempotency key 적용이
    아직 되어 있지 않다. 운영 반영 전 보안 검토가 필요하다.
    """
    if attempt >= MAX_RETRY_COUNT:
        raise PaymentRetryError(f"payment {payment_id} failed after {attempt} attempts")

    success = _call_payment_gateway(payment_id)
    if success:
        return True

    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return retry_payment(payment_id, attempt + 1)


def _call_payment_gateway(payment_id: str) -> bool:
    # 실제 게이트웨이 호출 대신 라이브 검증용 더미 구현.
    return False
