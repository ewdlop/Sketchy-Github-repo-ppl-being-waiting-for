# 👻 Phantom Read SQL - The Ghost in Your Transactions

**Phantom Read（幻讀）** 是資料庫事務隔離性問題中最詭異的現象之一──就像萬聖節的幽靈，明明剛才不在那裡的資料，再看一次時卻突然「出現」了！

---

## 🎃 什麼是 Phantom Read？

**Phantom Read** 發生在一個事務（Transaction）中：
1. 第一次查詢時，讀取了某個範圍的資料
2. 另一個事務在這個範圍內 **插入了新資料** 並提交
3. 第一個事務再次執行相同查詢時，發現了「幽靈般」的新資料

這就像你在一間空房間裡轉身，再轉回來時突然發現多了一個人！👻

---

## 🧛‍♂️ 實際範例：幽靈訂單

### Session 1（第一個事務）

```sql
-- 開始事務
BEGIN TRANSACTION;

-- 第一次查詢：檢查今天的訂單
SELECT * FROM orders
WHERE order_date = '2025-10-31'
  AND status = 'pending';

-- 假設結果：找到 3 筆訂單
-- order_id: 101, 102, 103
```

### Session 2（另一個事務）

```sql
-- 同時，另一個使用者下了新訂單
BEGIN TRANSACTION;

INSERT INTO orders (order_id, order_date, status)
VALUES (104, '2025-10-31', 'pending');

COMMIT;
-- 新訂單已提交！
```

### Session 1（繼續執行）

```sql
-- 再次執行相同查詢
SELECT * FROM orders
WHERE order_date = '2025-10-31'
  AND status = 'pending';

-- 👻 現在發現 4 筆訂單！
-- order_id: 101, 102, 103, 104（幽靈出現了！）

COMMIT;
```

---

## 🕷️ Phantom Read vs. 其他問題

### 對照表

| 現象 | 說明 | 萬聖節比喻 |
|------|------|-----------|
| **Dirty Read** | 讀到未提交的資料 | 🧟‍♂️ 殭屍資料（還沒完全死掉） |
| **Non-Repeatable Read** | 同一資料兩次讀取結果不同 | 🧙‍♀️ 變形怪（同個東西變了） |
| **Phantom Read** | 範圍查詢出現新資料 | 👻 幽靈（憑空出現） |

---

## 🪦 如何防止 Phantom Read？

### 方法 1：使用 `SERIALIZABLE` 隔離級別

```sql
-- 設定最高隔離級別
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

BEGIN TRANSACTION;

SELECT * FROM orders
WHERE order_date = '2025-10-31'
  AND status = 'pending';

-- 此時其他事務無法插入符合條件的新資料
-- 直到這個事務結束

COMMIT;
```

**說明**：
`SERIALIZABLE` 會鎖定查詢範圍，防止其他事務插入新資料。
但代價是性能較低，像是把整個房間都封鎖了。

---

### 方法 2：使用 `REPEATABLE READ` + 範圍鎖定

```sql
-- MySQL/InnoDB 的 REPEATABLE READ 可以防止 Phantom Read
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

BEGIN TRANSACTION;

SELECT * FROM orders
WHERE order_date = '2025-10-31'
  AND status = 'pending'
FOR UPDATE;  -- 明確鎖定範圍

-- 進行其他操作...

COMMIT;
```

---

### 方法 3：應用層邏輯處理

```sql
-- 使用快照讀取（某些資料庫支援）
BEGIN TRANSACTION;

-- 建立快照
CREATE TABLE temp_orders AS
SELECT * FROM orders
WHERE order_date = '2025-10-31';

-- 之後只操作快照資料
SELECT * FROM temp_orders;

-- 清理
DROP TABLE temp_orders;

COMMIT;
```

---

## 🧙‍♀️ 各資料庫的隔離級別對照

| 隔離級別 | Dirty Read | Non-Repeatable Read | Phantom Read |
|---------|-----------|-------------------|-------------|
| **READ UNCOMMITTED** | ✅ 可能 | ✅ 可能 | ✅ 可能 |
| **READ COMMITTED** | ❌ 防止 | ✅ 可能 | ✅ 可能 |
| **REPEATABLE READ** | ❌ 防止 | ❌ 防止 | ✅ 可能* |
| **SERIALIZABLE** | ❌ 防止 | ❌ 防止 | ❌ 防止 |

\* **注意**：MySQL InnoDB 的 `REPEATABLE READ` 可以防止 Phantom Read（使用 Next-Key Locks）

---

## 🩸 實戰範例：銀行轉帳系統

```sql
-- 糟糕的做法（可能出現 Phantom Read）
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

BEGIN TRANSACTION;

-- 計算當日總交易額
SELECT SUM(amount) AS total
FROM transactions
WHERE transaction_date = CURRENT_DATE;

-- 🚨 此時其他人可能插入新交易！

-- 基於上面的總額做決策
-- 可能會因為 Phantom Read 導致錯誤判斷

COMMIT;
```

```sql
-- 正確的做法
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

BEGIN TRANSACTION;

-- 計算當日總交易額（鎖定範圍）
SELECT SUM(amount) AS total
FROM transactions
WHERE transaction_date = CURRENT_DATE;

-- 安全：其他事務無法插入新交易

-- 基於準確的總額做決策
-- ...

COMMIT;
```

---

## 🎃 測試 Phantom Read

### 建立測試表

```sql
CREATE TABLE ghost_sightings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    location VARCHAR(100),
    sighting_time DATETIME,
    ghost_type VARCHAR(50)
);

-- 插入初始資料
INSERT INTO ghost_sightings (location, sighting_time, ghost_type)
VALUES 
    ('Haunted House', '2025-10-31 20:00:00', 'White Lady'),
    ('Cemetery', '2025-10-31 21:00:00', 'Skeleton'),
    ('Old Church', '2025-10-31 22:00:00', 'Poltergeist');
```

### 測試腳本（需要兩個 Session）

**Session 1:**
```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN TRANSACTION;

SELECT COUNT(*) AS ghost_count
FROM ghost_sightings
WHERE sighting_time >= '2025-10-31 20:00:00';
-- 結果：3

-- 等待 Session 2 插入資料...
```

**Session 2:**
```sql
BEGIN TRANSACTION;

INSERT INTO ghost_sightings (location, sighting_time, ghost_type)
VALUES ('Dark Forest', '2025-10-31 23:00:00', 'Phantom');

COMMIT;
```

**Session 1（繼續）:**
```sql
-- 再次查詢
SELECT COUNT(*) AS ghost_count
FROM ghost_sightings
WHERE sighting_time >= '2025-10-31 20:00:00';
-- 👻 結果：4（出現 Phantom Read！）

COMMIT;
```

---

## 🕸️ 性能 vs. 一致性的取捨

```
┌──────────────────────────────────────────────┐
│ Isolation Level Performance vs. Consistency  │
├──────────────────────────────────────────────┤
│                                              │
│ READ UNCOMMITTED   ████████████ (最快)      │
│ READ COMMITTED     ████████     (快)        │
│ REPEATABLE READ    █████        (中)        │
│ SERIALIZABLE       ██           (慢)        │
│                                              │
│ 一致性相反（越慢越一致）                      │
└──────────────────────────────────────────────┘
```

---

## 🦇 最佳實踐建議

1. **預設使用 `READ COMMITTED`**
   - 適合大多數應用場景
   - 性能與一致性平衡

2. **關鍵業務使用 `SERIALIZABLE`**
   - 銀行轉帳、庫存管理等
   - 需要完全一致性的操作

3. **使用樂觀鎖或版本控制**
   ```sql
   UPDATE orders
   SET status = 'processed', version = version + 1
   WHERE order_id = 123
     AND version = 1;  -- 只在版本匹配時更新
   ```

4. **避免長事務**
   - 減少鎖定時間
   - 降低 Phantom Read 風險

5. **監控與日誌**
   ```sql
   -- 記錄事務隔離級別
   SELECT @@transaction_isolation;
   
   -- 監控鎖等待
   SHOW ENGINE INNODB STATUS;
   ```

---

## 📚 參考資料

* [ANSI SQL-92 Transaction Isolation Levels](https://www.contrib.andrew.cmu.edu/~shadow/sql/sql1992.txt)
* [MySQL InnoDB Locking](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html)
* [PostgreSQL Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html)
* [SQL Server Isolation Levels](https://learn.microsoft.com/en-us/sql/t-sql/statements/set-transaction-isolation-level-transact-sql)

---

## 🎃 小測驗

**Q1**: 哪種隔離級別可以完全防止 Phantom Read？
<details>
<summary>點擊查看答案</summary>

**A**: `SERIALIZABLE` 隔離級別可以完全防止 Phantom Read。

</details>

**Q2**: Phantom Read 與 Non-Repeatable Read 的主要差異是什麼？
<details>
<summary>點擊查看答案</summary>

**A**: 
- **Non-Repeatable Read**: 同一筆資料兩次讀取時「內容改變」
- **Phantom Read**: 範圍查詢時「出現新資料」（資料筆數改變）

</details>

**Q3**: MySQL InnoDB 的 `REPEATABLE READ` 能防止 Phantom Read 嗎？
<details>
<summary>點擊查看答案</summary>

**A**: 能！MySQL InnoDB 使用 Next-Key Locks，在 `REPEATABLE READ` 級別就能防止 Phantom Read。這是 MySQL 的特殊實作，不同於標準 SQL。

</details>

---

👻 **Happy Halloween Coding!** 👻

記住：資料庫的幽靈不會害人，但 Phantom Read 會讓你的系統出 Bug！
