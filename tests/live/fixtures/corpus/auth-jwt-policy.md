# 인증/JWT 정책

## 토큰 발급 방식

로그인 성공 시 서버는 Access Token과 Refresh Token을 발급합니다. 서명 알고리즘은
**RS256**을 사용하며, 대칭키(HS256)는 사용하지 않습니다.

## 만료 시간

- Access Token 만료 시간: **30분**
- Refresh Token 만료 시간: **14일**

Access Token 만료 시 클라이언트는 Refresh Token으로 재발급 엔드포인트
(`POST /api/v1/auth/refresh`)를 호출해 새 Access Token을 받습니다.

## 시크릿 키 로테이션

서명에 사용하는 개인키는 **90일 주기**로 로테이션합니다. 로테이션 시 이전 키는
14일간 검증용으로만 유지되어, 로테이션 시점에 발급된 토큰도 만료 전까지 정상
검증됩니다.

## 클레임 구성

JWT payload에는 `sub`(사용자 ID), `role`, `workspace_ids`, `iat`, `exp` 클레임이
포함됩니다. 비밀번호, 이메일 등 민감 정보는 클레임에 포함하지 않습니다.

## 토큰 폐기

로그아웃 시 Refresh Token은 서버 측 블랙리스트에 등록되어 즉시 무효화됩니다.
Access Token은 만료 전까지는 별도 폐기 없이 자연 만료됩니다(무상태 검증 특성).
