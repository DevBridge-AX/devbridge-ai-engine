-- =============================================================================
-- DevBridge AI Engine - DB 초기화 (alembic upgrade head 실행 전 1회 필요)
--
-- 목적: AI 도메인 전용 스키마와, 해당 스키마에만 권한을 가진 AI 엔진 전용 계정을 만든다.
--       Spring 백엔드 계정/스키마와 분리하기 위함이다.
--
-- 플레이스홀더 (.env.example의 변수명과 동일, 실행 전 실제 값으로 치환):
--   ${MYSQL_USER_AI}      AI 엔진 DB 계정명        (.env의 MYSQL_USER_AI)
--   ${MYSQL_PASSWORD_AI}  AI 엔진 DB 계정 비밀번호  (.env의 MYSQL_PASSWORD_AI)
--   devbridge_ai          스키마명. .env의 DATABASE_URL 끝 경로(/devbridge_ai)와 일치해야 한다.
--
-- 실행 예 (root 등 CREATE USER 권한이 있는 계정으로):
--   set -a; . ./.env; set +a
--   envsubst '${MYSQL_USER_AI} ${MYSQL_PASSWORD_AI}' < init-db/init.sql | mysql -u root -p
--
-- 재실행해도 안전하도록 IF NOT EXISTS를 사용한다.
-- 주의: 이 파일은 스키마/계정만 만든다. 테이블은 alembic이 만든다.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS devbridge_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS '${MYSQL_USER_AI}'@'%' IDENTIFIED BY '${MYSQL_PASSWORD_AI}';

-- 권한은 AI 스키마로만 한정한다. 다른 스키마(Spring 소유)에는 권한을 주지 않는다.
GRANT ALL PRIVILEGES ON devbridge_ai.* TO '${MYSQL_USER_AI}'@'%';

FLUSH PRIVILEGES;
