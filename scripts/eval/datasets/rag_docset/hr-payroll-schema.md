# 인사 급여 테이블 스키마 (제한 문서)

인사 시스템의 급여 관련 테이블 정의입니다. 접근 권한이 있는 인원만 열람할 수 있습니다.

## PAYROLL_MASTER
- `employee_no` VARCHAR(10) PK, `base_salary` DECIMAL(12,0), `grade_code` CHAR(2), `bank_account_enc` VARBINARY(256)
- `bank_account_enc`는 AES-256으로 암호화해 저장합니다.

## PAYROLL_MONTHLY
- `employee_no`, `pay_month` CHAR(6), `gross_pay`, `tax_amount`, `net_pay`
- 급여 확정은 매월 20일 18:00에 마감되며, 마감 후에는 수정할 수 없고 정정 전표로만 보정합니다.

## 접근 규칙
- 조회는 인사팀 담당자 A와 재무팀 담당자 B 두 그룹만 가능합니다.
- 모든 조회 이력은 감사 로그에 남깁니다.
