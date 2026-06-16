"""
앱 전역 설정.

환경 변수(.env)를 로드하여 DB, 벡터스토어, 임베딩, LLM provider, Spring 연동 등에
필요한 설정값을 제공합니다. pydantic-settings의 BaseSettings를 사용합니다.

TODO:
- 운영/스테이징/로컬 환경별 설정 분리 전략 검토
- LLM provider별 설정 스키마 세분화 (core/llm/provider.py와 함께 설계)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    # Internal API 인증 (Spring -> FastAPI, X-Internal-Api-Key 헤더)
    internal_api_key: str = "changeme"

    # Database (AI 도메인 전용 스키마, Spring과 별도 DB 계정)
    database_url: str = "mysql+pymysql://ai_engine_user:changeme@localhost:3306/devbridge_ai"

    # Vector store (임베디드/파일 기반, 별도 서버 없음)
    vector_store_provider: str = "chroma"
    vector_store_path: str = "./data/vector_store"

    # Embeddings (현재: Gemini Embedding via GMS 프록시)
    # embedding_api_base_url + embedding_model 조합으로 엔드포인트 URL을 구성합니다.
    embedding_api_base_url: str = "https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta/models"
    embedding_model: str = "gemini-embedding-2"
    embedding_model_version: str = "v1"
    embedding_api_key: str = ""  # GMS_KEY

    # LLM provider (core/llm/provider.py 추상 인터페이스 뒤에서 사용)
    anthropic_api_key: str = ""
    main_model: str = "claude-sonnet-4-6"
    rewrite_model: str = "claude-haiku-4-5-20251001"

    # provider.py 컨텍스트 길이 가드
    max_context_tokens: int = 30000

    # Spring backend(backend 레포) 연동
    spring_backend_base_url: str = "http://localhost:8080"
    spring_user_lookup_path: str = "/internal/users/lookup"


@lru_cache
def get_settings() -> Settings:
    """Settings 싱글톤 인스턴스를 반환합니다."""
    return Settings()
