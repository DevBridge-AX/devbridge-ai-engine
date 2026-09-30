# 결제 API

## 개요

결제 API는 외부 결제 게이트웨이인 **TossPayments**와 연동하여 카드/계좌이체 결제를
처리합니다.

## 소스 코드 위치

결제 API의 소스 코드는 아래 경로에 위치합니다.

```
backend/payment-api/src/main/java/com/acme/payment
```

주요 클래스는 `PaymentController`(엔드포인트), `PaymentGatewayClient`(TossPayments
연동), `PaymentTransactionRepository`(거래 기록 저장)입니다.

## 주요 엔드포인트

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| POST | /api/v1/payments | 결제 요청 생성 |
| GET | /api/v1/payments/{id} | 결제 상태 조회 |
| POST | /api/v1/payments/{id}/cancel | 결제 취소 |

## 타임아웃 및 재시도

TossPayments 게이트웨이 호출 타임아웃은 30초이며, 타임아웃 발생 시 최대 2회까지
지수 백오프로 재시도합니다. 재시도 후에도 실패하면 `PAYMENT_GATEWAY_TIMEOUT` 오류
코드를 반환합니다.

## 멱등성

결제 요청 생성 시 클라이언트가 `Idempotency-Key` 헤더를 반드시 포함해야 하며, 동일
키로 재요청 시 새 결제를 생성하지 않고 기존 결제 결과를 그대로 반환합니다.
