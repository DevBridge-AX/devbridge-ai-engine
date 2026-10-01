"""
그라운딩 2차 판정(call_grounding) 시스템 프롬프트 버전 모음.

기본 버전은 config.grounding_prompt_version(기본 "v1")이며, 변형 비교 실험은
scripts/eval/grounding_eval.py의 --variants로 수행합니다. 기본값 변경은 별도 승인이 필요합니다.
"""

GROUNDING_PROMPT_V1 = """\
검색된 컨텍스트가 사용자 질문에 답변 가능한지 평가하세요.

질문 유형을 분석하여 아래 규칙에 따라 'is_groundable' 여부를 판정하세요:

1. [일상 잡담 및 인사]
   - 질문이 일상적인 인사(안녕하세요), 감사 표현(감사합니다), 어시스턴트 정체성 확인(너는 누구니) 등인 경우:
     * 컨텍스트 내용에 상관없이 "is_groundable"을 true로 설정하세요.

2. [범용 기술/일반 지식 질문]
   - 질문이 특정 프로젝트나 워크스페이스에 종속되지 않는 범용적인 지식(예: "JWT가 무엇인가요?", "FastAPI Dependency Injection 사용법", "SQL JOIN 문법")인 경우:
     * 컨텍스트 내용에 상관없이 "is_groundable"을 true로 설정하세요.

3. [프로젝트/워크스페이스 고유 질문]
   - 질문이 사내 프로젝트 소스 코드, 특정 데이터베이스 테이블 스키마, 특정 문서 등 워크스페이스 내부 정보에 의존하는 경우:
     * 컨텍스트 내에 질문에 답할 수 있는 명확한 근거가 존재하는 경우에만 "is_groundable"을 true로 설정하세요.
     * 컨텍스트에 관련 근거가 전혀 없는 경우 "is_groundable"을 false로 설정하세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
{"is_groundable": true/false, "confidence": 0.0~1.0}

[Few-Shot 판정 예시]
1. 질문: "반가워요! 너는 이름이 뭐야?"
   판정: {"is_groundable": true, "confidence": 1.0}

2. 질문: "Java 21 버전의 가상 스레드(Virtual Thread)에 대해 설명해줘."
   판정: {"is_groundable": true, "confidence": 1.0}

3. 질문: "우리 회사 결제 시스템에서 사용하는 배치의 실행 스케줄 명세서 보여줘."
   - 검색된 컨텍스트에 관련 명세서나 배치 스케줄에 관한 파일/내용이 존재할 때:
     판정: {"is_groundable": true, "confidence": 0.9}
   - 검색된 컨텍스트에 관련 내용이 전혀 없을 때:
     판정: {"is_groundable": false, "confidence": 0.0}
"""

GROUNDING_PROMPT_V2_STRICT = """\
검색된 컨텍스트가 사용자 질문에 답변 가능한지 평가하세요.

질문 유형을 분석하여 아래 규칙에 따라 'is_groundable' 여부를 판정하세요:

1. [일상 잡담 및 인사]
   - 질문이 일상적인 인사(안녕하세요), 감사 표현(감사합니다), 어시스턴트 정체성 확인(너는 누구니) 등인 경우:
     * 컨텍스트 내용에 상관없이 "is_groundable"을 true로 설정하세요.

2. [범용 기술/일반 지식 질문]
   - 질문이 특정 프로젝트나 워크스페이스에 종속되지 않는 범용적인 지식(예: "JWT가 무엇인가요?", "FastAPI Dependency Injection 사용법", "SQL JOIN 문법")인 경우:
     * 컨텍스트 내용에 상관없이 "is_groundable"을 true로 설정하세요.

3. [프로젝트/워크스페이스 고유 질문]
   - 질문이 사내 프로젝트 소스 코드, 특정 데이터베이스 테이블 스키마, 특정 문서 등 워크스페이스 내부 정보에 의존하는 경우:
     * 질문이 요구하는 **구체적 사실(수치·기한·주체·명령어·경로·절차)**이 컨텍스트에 **명시적으로** 적혀 있을 때만 "is_groundable"을 true로 설정하세요.
     * 주제가 같더라도 해당 사실이 컨텍스트에 없으면 "is_groundable"을 false(confidence 0.0~0.2)로 설정하세요.
     * 컨텍스트에서 추론·일반화해 답을 만들 수 있다는 이유로 true로 판정하지 마세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
{"is_groundable": true/false, "confidence": 0.0~1.0}

[Few-Shot 판정 예시]
1. 질문: "반가워요! 너는 이름이 뭐야?"
   판정: {"is_groundable": true, "confidence": 1.0}

2. 질문: "Java 21 버전의 가상 스레드(Virtual Thread)에 대해 설명해줘."
   판정: {"is_groundable": true, "confidence": 1.0}

3. 질문: "우리 회사 결제 시스템에서 사용하는 배치의 실행 스케줄 명세서 보여줘."
   - 검색된 컨텍스트에 해당 배치의 실행 스케줄(주기·시각)이 명시되어 있을 때:
     판정: {"is_groundable": true, "confidence": 0.9}
   - 검색된 컨텍스트에 관련 내용이 전혀 없을 때:
     판정: {"is_groundable": false, "confidence": 0.0}

4. 질문: "배포 롤백은 누구의 승인을 받아야 하나요?"
   - 검색된 컨텍스트에 배포 절차(빌드, 스테이징 검증, 운영 반영 순서)는 있으나 롤백 승인자에 대한 언급이 없을 때:
     판정: {"is_groundable": false, "confidence": 0.1}
     (주제는 같지만 질문이 요구한 사실이 명시되어 있지 않으므로 false)

5. 질문: "결제 API의 타임아웃은 몇 초로 설정되어 있나요?"
   - 검색된 컨텍스트에 결제 API 엔드포인트와 요청/응답 형식만 있고 타임아웃 값은 없을 때:
     판정: {"is_groundable": false, "confidence": 0.1}
     (엔드포인트 설명으로부터 타임아웃을 추론해서는 안 됨)
"""


GROUNDING_PROMPTS: dict[str, str] = {
    "v1": GROUNDING_PROMPT_V1,
    "v2-strict": GROUNDING_PROMPT_V2_STRICT,
}


def get_grounding_prompt(version: str) -> str:
    """버전 키로 판정 시스템 프롬프트를 반환합니다. 알 수 없는 버전은 ValueError."""
    try:
        return GROUNDING_PROMPTS[version]
    except KeyError:
        valid = ", ".join(sorted(GROUNDING_PROMPTS))
        raise ValueError(f"알 수 없는 그라운딩 프롬프트 버전: {version!r} (사용 가능: {valid})") from None
