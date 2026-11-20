# 任务 6B：NG 图片留存

- 完成日期：2026-11-20
- 对应 SRS：FR-013、§8.3
- 入口：单图 `--db` 模式
  - `--ng-image-dir <dir>`：启用 NG 图片留存
  - `--ng-image-mode original|annotated`：留存原图或 5C 标注图
  - 默认关闭；只有同时指定 `--db` 和单图 `--image` 时可用

## 1. 留存规则

- 默认关闭：不指定 `--ng-image-dir` 时完全不写图片，`image_path` 为 NULL。
- 只保存 **NG** 帧；OK 帧不写图片，也不创建目录。
- `original`：按字节复制输入文件（`std::filesystem::copy_file`）。
- `annotated`：复用 5C `annotateFrame()`，保存为 PNG。
- 文件名以事件 UUID 为前缀：
  - `original`：`<event_uid><原扩展名>`
  - `annotated`：`<event_uid>.png`
  - 若目标已存在，依次尝试 `<uid>-1`、`<uid>-2` …，不覆盖已有文件。
- `--ng-image-dir` 在启动时解析为真实绝对路径，并拒绝已存在但不是目录的
  路径；图片路径也是绝对路径，查询进程可在任意工作目录定位。
- 图片路径会做与模型、输入图片、JSON/标注输出和数据库文件的精确冲突检查。

## 2. 入库与查询

执行顺序：

1. 计算 OK/NG 和事件 UUID。
2. NG 且启用留存时，先写图片。
3. 打开数据库并写入事件；成功时 `image_path` 为已写出的可定位路径。
4. 查询进程输出 `image=<path>`；未留存时为 `image=-`。

单图成功示例：

```text
quality: NG (confidence>0.25, qualifying=1/1)
event saved: uid=d2674773-... db=/.../events.db image=/.../original/d2674773-....png
```

新进程查询：

```text
event uid=d2674773-... product=NG-ORIG status=NG defects=1
  image=/.../original/d2674773-....png model=...
```

图片以事件 ID 命名，且 `image=` 路径可直接打开。

## 3. 失败处理

- **图片写入失败**（目录不可创建、复制失败、标注写失败）：
  - 仍写入 NG 事件，`image_path` 为 NULL；
  - stderr 记录事件 ID 和失败原因；
  - CLI 退出码 1；
  - 查询仍能查到该 NG 事件（`image=-`）。
- **图片已写出但数据库打开/写入失败**：
  - 删除本次新写出的图片；
  - stderr 记录事件 ID、数据库错误和清理结果；
  - 若删除失败，额外记录 `failed to remove retained image ...`；
  - CLI 退出码 1。
- 数据库写入失败不会改变 OK/NG 判定；可选 JSON/标注/窗口输出的失败
  继续独立报告。

## 4. 验收证据

完整日志：`docs/ng-image-retention-run.txt`。

- `original`：保存文件大小与输入一致（4461 bytes），SHA256 相同；
- `annotated`：保存为 320×200 PNG；
- 默认关闭：查询输出 `image=-`；
- OK + 启用目录：目录中 0 个文件，事件 `status=OK image=-`；
- 图片写失败：退出 1，NG 事件仍可查，`image=-`；
- 数据库失败：图片先写出后被删除，目录 0 个文件，日志包含
  `removed retained image ... after database failure`。

## 5. 测试

```bash
ctest --test-dir build --output-on-failure
```

本次完整套件 **82/82 通过**；无模型/图片/摄像头时 79 个测试通过。

任务 6B 新增 `cli_ng_image`，覆盖：

| 场景 | 断言 |
| --- | --- |
| 默认关闭 | NG 事件 `image=-`，不写图 |
| `original` | 文件存在、以事件 ID 命名、SHA256 与输入一致 |
| `annotated` | 文件存在、以事件 ID 命名、320×200 PNG |
| OK 帧 | 目录中 0 个文件，事件 `image=-` |
| 图片写失败 | 退出 1，日志含事件 ID，NG 事件仍可查且 `image=-` |
| 入库失败 | 已写图片被删除，日志记录清理，目录 0 个文件 |
| 选项守卫 | 缺 `--db`、缺目录、非法模式返回 2 |

## 6. 限制与待确认

1. 仅单图 `--db` 模式；目录批量和摄像头的留存未接。
2. 只留存 NG；没有 OK 图片留存策略。
3. 没有容量上限、保留天数或自动清理策略；文件名保证不覆盖，但目录会
   持续增长。
4. 图片路径存绝对路径；移动数据库/图片目录后需人工迁移。
5. AT-009 真实 USB 摄像头、AT-015 真实桌面交互仍待实机。
