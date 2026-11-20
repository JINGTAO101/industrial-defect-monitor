# 任务 6A：SQLite 检测事件保存与查询

- 完成日期：2026-11-20
- 对应 SRS：FR-012、NFR-011；AT-010
- 入口：
  - 写入：`edge_ai --model ... --image ... --db events.db [--product-id ID]`
  - 查询：`edge_ai --query --db events.db [--from T] [--to T] [--status OK|NG] [--product-id ID]`
- 实现：`include/detection_store.hpp`、`src/detection_store.cpp`

## 1. 依赖与范围

- 使用系统 SQLite3 开发库；Ubuntu/Debian 安装：
  ```bash
  sudo apt install libsqlite3-dev
  ```
- 本阶段只接**单图模式**：必须显式指定 `--db` 才写库；目录批量和摄像头
  暂不支持数据库参数（CLI 返回 2）。
- `--db` 只能是持久化文件路径；`:memory:` 和 `file:` URI 形式直接返回 2，
  避免“保存成功、进程退出后查不到”。
- 查询是独立入口：`--query` 不需要模型/图片，只打开指定数据库。
- 数据库旁会生成 SQLite 的 `-journal`/`-wal`/`-shm` 文件；这些路径也纳入
  输出冲突检查，不能被 `--json-out`/`--dump-input`/`--save-annotated`
  指向，也不能与模型或输入图片重名。
- `--db` 先解析符号链接到真实路径，再用同一真实路径构造旁文件名并打开
  SQLite。`alias.db -> events.db` 时检查的是 `events.db-journal`，而不是
  `alias.db-journal`。

## 2. 数据库结构

```sql
CREATE TABLE detection_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid            TEXT NOT NULL UNIQUE,
    occurred_at_ms       INTEGER NOT NULL,
    source               TEXT NOT NULL,
    product_id           TEXT,
    is_ng                INTEGER NOT NULL CHECK (is_ng IN (0,1)),
    defect_count         INTEGER NOT NULL CHECK (defect_count >= 0),
    model_identifier     TEXT NOT NULL,
    model_path           TEXT,
    image_path           TEXT,
    confidence_threshold REAL NOT NULL,
    created_at_ms        INTEGER NOT NULL
);

CREATE TABLE detection_defects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    INTEGER NOT NULL REFERENCES detection_events(id) ON DELETE CASCADE,
    class_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
);
```

- `event_uid`：从 `random_device` 取 256 位种子初始化线程局部生成器，每次
  生成 128 位 UUID v4 形状 ID，并混入进程 ID、时钟和单调计数；有 UNIQUE
  约束。旧的单 32 位种子实现已在大批量生成中观测到碰撞。
- `schema_meta.schema_version` 在打开时校验。新库在一个事务里原子创建表、
  索引和版本行；已存在的库若不是版本 `1`，写入和查询都在任何 DDL/写入前
  拒绝。
- 没有 `schema_meta` 但已经存在用户 schema 对象的旧库**不会**被当作新库
  初始化：事务内用 `name NOT GLOB 'sqlite_*'` 检查 `sqlite_master` 中的
  表、视图、索引和触发器（`GLOB` 的 `_` 是字面字符；`LIKE` 的 `_` 会匹配
  任意单字符，曾把合法表名 `sqliteNotes` 误判为内部表）。只要有任何用户
  对象就拒绝并回滚，文件保持原样。仅含视图的旧库同样会被拒绝。
- `occurred_at_ms`：输入帧成功解码时的 `system_clock` 墙钟时间（epoch ms）。
- `source`：输入来源（单图路径）。
- `product_id`：`--product-id`；未提供时写 NULL。
- `is_ng` / `defect_count`：质量判定结果和**达到阈值**的缺陷数。
- `model_identifier`：模型 SHA256；`model_path` 为加载路径。
- `image_path`：本阶段允许为空（NULL），留给 FR-013 图片留存。
- 缺陷明细保存类别 ID/名称、置信度、后处理还原后的**原图坐标**。
- 索引覆盖时间、OK/NG、产品 ID 和明细的 `event_id`。

## 3. 写入与事务

- 仅当 `--db` 非空时写库；数据库文件不存在则创建，父目录不存在则创建。
- 事件主记录和全部明细在同一个 `BEGIN IMMEDIATE ... COMMIT` 事务中写入。
- 事件插入失败、任一明细插入失败、或 COMMIT 失败都会 `ROLLBACK`；
  不会留下半条事件。
- 显式指定 `--db` 时，**先持久化事件**，再做 `--json-out`/标注/窗口等
  可选输出。可选导出失败会单独报告并把退出码置为 1，但已完成的检测事件
  仍在数据库中；写库失败也不会阻止后续导出尝试。
- 写入失败时：
  - 检测结果仍按模型输出打印和判定（写库失败不会被改写成 OK/NG）；
  - stderr 输出 `event database ... failed: ...`；
  - 数据库打开或事件插入失败时日志包含 `event uid '<id>'`
    （SRS FR-012/§10）；
  - 退出码为 1。
- 成功时 stdout 输出 `event saved: uid=... db=...`。

## 4. 查询

- `--query --db <path>` 以只读方式打开数据库；数据库不存在或没有
  `detection_events`/`detection_defects` 表时报错并返回 1。
- 过滤条件可单独或组合，全部为 AND：
  - `--from <time>`：`occurred_at_ms >= ?`（含）
  - `--to <time>`：`occurred_at_ms <= ?`（含）
  - `--status OK|NG`：`is_ng = ?`
  - `--product-id <id>`：`product_id = ?`
- 所有条件、事件 ID 和插入值都使用 `sqlite3_bind_*` 参数绑定，不拼接
  用户输入到 SQL。
- `<time>` 支持两种形式：
  - epoch 毫秒整数，例如 `1763600000000`；
  - ISO-8601 UTC，例如 `2026-11-20T10:00:00Z`（可带 `.mmm`）。
  - ISO 日期逐字段严格校验：月份 1-12、日期不超过当月天数（含闰年）、
    时 0-23、分/秒 0-59；`2026-02-31` 等非法值返回 2，不会被
    `std::get_time` 归一化。仅接受 UTC（`Z` 或 `±00:00`）。
- 结果按 `occurred_at_ms ASC, id ASC` 排序；输出事件字段和每条缺陷明细，
  时间同时显示为 ISO 字符串。
- 无匹配记录是正常结果：输出 `events: total=0`，退出码 0。

## 5. 验收证据

完整日志：`docs/detection-store-run.txt`。

固定样例（合成 ONNX，不含训练权重）：

- OK：`zero_output.onnx` → 0 检测；
- NG：`fixed_detection.onnx` → 1 个 `crazing`，置信度 0.9，
  `bbox_xyxy=[40,30,160,120]`。

写入两个新进程后，用新的查询进程读到：

```text
events: total=2
event uid=... product=PROD-OK status=OK defects=0 image=- model=227227...
event uid=... product=PROD-NG status=NG defects=1 image=- model=7848e5...
  defect class_id=0 class=crazing confidence=0.900000 bbox=[40, 30, 160, 120]
```

筛选：

```text
$ edge_ai --query --db events.db --status NG
events: total=1
event ... product=PROD-NG status=NG defects=1 image=-
  defect class_id=0 class=crazing confidence=0.900000 bbox=[40, 30, 160, 120]

$ edge_ai --query --db events.db --product-id PROD-OK
events: total=1
event ... product=PROD-OK status=OK defects=0 image=-
```

CLI 测试还用两个相隔 1.1 秒的写入验证 `--from`/`--to` 各只选中一条，
并验证组合筛选。

写库失败：

```text
$ edge_ai --model ... --image ... --db non_square_320x200.png/events.db
...
quality: OK (confidence>0.25, qualifying=0/0)
edge_ai: event database open failed for event uid '<id>': cannot create
database directory 'tests/fixtures/non_square_320x200.png': Not a directory
# exit code: 1
```

质量判定已输出，错误只来自数据库。

明细失败回滚：单元测试在数据库上创建
`BEFORE INSERT ON detection_defects ... RAISE(ABORT, 'injected defect failure')`
触发器，插入含两个缺陷的事件。第二个明细失败后，查询主表为 0 行，
证明主记录与第一个明细都被回滚。

无版本旧库：`legacy_no_meta.db`（旧 `detection_events` 表）、
`legacy_sqlite_notes.db`（用户表 `sqliteNotes`）和
`legacy_view_only.db`（只有视图）均在初始化前被拒绝；三者的 SHA256
前后一致，且没有留下 `-journal`。

## 6. 测试

```bash
ctest --test-dir build --output-on-failure
```

本次完整套件 **85/85 通过**；无模型/图片/摄像头时 82 个测试通过。

任务 6A 新增：

| 测试 | 内容 |
| --- | --- |
| `storage_roundtrip` | 事件+明细写入、唯一 ID、字段读取、关闭后重开仍存在 |
| `storage_rollback_on_defect_failure` | 明细触发器失败后主记录回滚为 0 行 |
| `storage_query_filters` | 状态/产品/时间范围/组合筛选、产品 ID 参数绑定 |
| `storage_event_uid_unique` | 连续生成 200,000 个 ID 无重复、格式正确 |
| `storage_timestamp_parsing` | 合法 epoch/ISO；闰年、非法日历日期、非法时区和格式被拒绝 |
| `storage_rejects_unknown_schema_version` | 版本 999 的库在读写前被拒绝且事件数不变 |
| `storage_rejects_legacy_database` | 有旧表但无 schema_meta 的库被拒绝，文件字节不变 |
| `cli_storage` | OK/NG 两次进程写入、新进程查询、筛选、写库失败、sidecar 冲突（含符号链接别名）、`:memory:`、非法日期、导出失败仍入库；`legacy_no_meta`/`legacy_sqlite_notes`/`legacy_view_only` 三种无版本旧库均退出 1、SHA256 不变、无 journal；选项守卫 |

## 7. 限制与待确认

1. 仅单图写入；目录批量和摄像头的持久化未接。
2. `image_path` 为空；FR-013 图片留存另行实现。
3. 只保存达到质量阈值的缺陷明细；低于阈值的候选不写库。
4. 数据库是单文件本地库，未做多进程并发写入压力测试。
5. 仅支持 schema 版本 1；升级/迁移未实现，未知版本会被拒绝。
6. 不支持 `:memory:`/URI 数据库；必须是持久化文件路径。
7. AT-010 的本机验收使用合成 fixture；真实产线数据量/长时间运行未测。
8. 查询输出为文本，未提供 JSON/分页。
