# 알림 서비스 API 명세

알림 서비스는 사용자에게 인앱 알림과 푸시 알림을 전달하는 내부 서비스입니다.

## 엔드포인트
- `POST /notifications` : 알림 1건을 생성합니다. 본문에 `user_id`, `type`, `message`를 담습니다.
- `GET /notifications?user_id=` : 미읽음 알림 목록을 최신순으로 최대 50건 반환합니다.
- `PATCH /notifications/{id}/read` : 알림을 읽음 처리합니다.

## 제약
- `message`는 최대 200자이며 초과하면 400 `MESSAGE_TOO_LONG`을 반환합니다.
- 동일 사용자에게 같은 `type`의 알림은 10분 안에 한 번만 생성되고, 중복 요청은 409 `DUPLICATE_NOTIFICATION`으로 거절됩니다.
- 푸시 전송 실패 시 최대 3회까지 재시도하며 재시도 간격은 30초입니다.

## 인증
- 모든 요청에는 내부 호출용 헤더 `X-Service-Token`이 필요합니다.
