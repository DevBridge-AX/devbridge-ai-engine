# 주문 DB 스키마

주문 도메인은 ORDERS, ORDER_ITEMS, ORDER_STATUS_HISTORY 세 테이블로 구성됩니다.

## ORDERS
- `id` BIGINT PK, `customer_id` BIGINT, `total_amount` DECIMAL(12,2), `status` VARCHAR(20), `ordered_at` DATETIME
- `status`는 CREATED, PAID, SHIPPED, CANCELED 중 하나입니다.
- `customer_id`에 인덱스 `idx_orders_customer`가 있습니다.

## ORDER_ITEMS
- `id` BIGINT PK, `order_id` BIGINT FK(ORDERS.id), `sku` VARCHAR(32), `quantity` INT, `unit_price` DECIMAL(10,2)
- 한 주문에는 최대 100개의 품목을 담을 수 있습니다.

## ORDER_STATUS_HISTORY
- 상태가 바뀔 때마다 `order_id`, `from_status`, `to_status`, `changed_at`을 한 행씩 추가하며, 수정과 삭제는 하지 않는 추가 전용 테이블입니다.
