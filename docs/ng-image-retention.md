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
- 写入先创建同目录的独占隐藏临时文件（`O_CREAT|O_EXCL`），完整写入、
  `fsync`、关闭后，用 `link(2)` 发布最终文件名；`link` 遇到已有目录项
  （包括悬空符号链接）会返回 `EEXIST`，因此不会跟随链接写入或覆盖。
  任一步失败都会删除本次临时文件，不会留下截断的最终图片。
- `--ng-image-dir` 解析为真实绝对路径；图片路径也是绝对路径，查询进程可在
  任意工作目录定位。
- 留存目录不得等于、包含或位于数据库文件或其
  `-journal`/`-wal`/`-shm` 旁文件之下（解析符号链接后比较），否则返回 2，
  不会把数据库文件变成目录。拒绝发生在创建目录和打开数据库之前，同一个
  数据库之后仍可正常写入。
- 留存目录已存在但是普通文件时，不在推理前报参数错误；写图阶段失败，
  NG 事件仍入库、`image_path` 为 NULL、日志含事件 ID、退出 1。
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

- **图片写入失败**（目录不可创建/是普通文件、复制失败、编码失败、
  写入/fsync/发布失败）：
  - 仍写入 NG 事件，`image_path` 为 NULL；
  - stderr 记录事件 ID 和失败原因；
  - CLI 退出码 1；
  - 查询仍能查到该 NG 事件（`image=-`）；
  - 临时文件被删除，不留下截断的最终图片。
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

本次完整套件 **85/85 通过**；无模型/图片/摄像头时 82 个测试通过。

任务 6B 新增：

| 测试 | 内容 |
| --- | --- |
| `ng_image_original_and_annotated` | 原图字节复制、标注 PNG、无临时文件残留 |
| `ng_image_dangling_symlink` | 悬空符号链接不被跟随/覆盖，自动使用 `-1` 名称 |
| `ng_image_failure_cleanup` | 限制文件大小导致写失败后，临时文件和最终图片都不存在 |
| `cli_ng_image` | 默认关闭、原图/标注、OK 不留图、写图失败仍入库、入库失败清理、数据库/旁文件/子路径冲突、符号链接别名、普通文件目录、选项守卫 |

`cli_ng_image` 还验证：留存目录等于数据库、等于 `-journal`/`-wal`/`-shm`、
位于数据库路径下或通过符号链接指向旁文件时均返回 2，未创建目录，且同一
数据库随后仍能正常写入和查询。

## 6. 限制与待确认

1. 仅单图 `--db` 模式；目录批量和摄像头的留存未接。
2. 只留存 NG；没有 OK 图片留存策略。
3. 没有容量上限、保留天数或自动清理策略；文件名保证不覆盖，但目录会
   持续增长。
4. 图片路径存绝对路径；移动数据库/图片目录后需人工迁移。
5. AT-009 真实 USB 摄像头、AT-015 真实桌面交互仍待实机。
