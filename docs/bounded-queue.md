# 任务 4A：可关闭的线程安全有界队列

- 完成日期：2026-11-18
- 对应 SRS：FR-010、NFR-002、NFR-003；AT-008
- 实现：`include/bounded_queue.hpp`（header-only 模板）
- 当前状态：队列及其独立 CTest 已完成；任务 4B 再把它接入采集/推理/事件
  三个线程。当前主程序仍是逐件同步处理，尚未使用该队列。

## 1. API

```cpp
enum class PopStatus { Item, Closed };

struct QueueStats {
    std::size_t enqueued;
    std::size_t dequeued;
    std::size_t max_depth;
    std::size_t producer_wait_count;  // push 因满而进入等待的次数
    std::size_t consumer_wait_count;  // pop 因空而进入等待的次数
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity);  // capacity == 0 抛 invalid_argument
    bool push(T value);                            // 满时阻塞；关闭后返回 false
    PopStatus pop(T& value);                       // 空时阻塞；关闭并排空后返回 Closed
    void close();                                  // 幂等，唤醒所有等待者
    std::size_t capacity() const;
    std::size_t size() const;
    bool isClosed() const;
    QueueStats stats() const;
};
```

## 2. 关闭语义

1. `close()` 可重复调用；每次都把所有阻塞的生产者/消费者唤醒。
2. 关闭后 `push()` 立即返回 `false`，不接收新元素；已经阻塞在满队列上的
   生产者被唤醒后同样返回 `false`。
3. 关闭前已成功入队的元素仍按 FIFO 顺序取完；消费者在排空后得到
   `PopStatus::Closed`，再次调用仍得到 `Closed`。
4. 队列应在析构前关闭并 join 所有使用者；析构本身不负责唤醒线程。
5. 容量为 0 在构造时用 `std::invalid_argument` 明确拒绝。

## 3. 测试

```bash
ctest --test-dir build -R '^queue_' --output-on-failure
```

6 个独立 CTest，均设置 `TIMEOUT 10`：

| 测试 | 内容 |
| --- | --- |
| `queue_reject_zero_capacity` | 容量 0 抛明确异常 |
| `queue_full_blocks` | 容量 2，A/B 后 C 阻塞；取走 A 后 C 入队；顺序 B、C；记录满载等待 |
| `queue_empty_blocks` | 消费者先阻塞；生产者放入后被唤醒并收到元素 |
| `queue_close_drains` | close 幂等；关闭后拒绝入队；已入队项排空后返回 Closed |
| `queue_close_wakes_waiters` | close 唤醒阻塞的生产者和消费者 |
| `queue_concurrent_no_loss` | 递增序号 + 慢消费者，2000 项无丢失/重复/乱序 |

## 4. 实测结果

满载场景（确定性）：

```text
full-block report: capacity=2 enqueued=3 dequeued=3 max_depth=2
                   producer_wait_count=1 consumer_wait_count=0
```

并发积压场景（等待次数受调度影响，本次运行）：

```text
concurrency report: capacity=4 enqueued=2000 dequeued=2000 max_depth=4
                    producer_wait_count=1973 consumer_wait_count=1
```

- 满载场景验证了 C 在 A 被消费前无法入队，取走 A 后 C 成功入队，剩余
  出队顺序为 B、C。
- 并发场景消费端严格收到 `0..1999`，无丢失、无重复、无乱序；
  `max_depth=4` 不超过容量；生产者发生过多次满载等待。
- 所有测试在 CTest 中超时上限为 10 秒，死锁不会永久挂住测试。

## 5. 限制

- 当前只测试单生产者/单消费者；多生产者/多消费者尚未专项验证。
- 尚无优先级、取消、超时 push/pop、批量 pop 等策略。
- 队列未接入主程序；4B 负责线程接线和端到端背压。
- 关闭后仍允许排空；不会丢弃已入队元素。
