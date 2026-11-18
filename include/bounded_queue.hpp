#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace edge_ai {

enum class PopStatus { Item, Closed };

struct QueueStats {
    std::size_t enqueued = 0;
    std::size_t dequeued = 0;
    std::size_t max_depth = 0;
    std::size_t producer_wait_count = 0;
    std::size_t consumer_wait_count = 0;
};

// Thread-safe bounded queue with blocking push/pop.
//
// Close semantics:
//  - close() is idempotent and wakes all blocked producers/consumers.
//  - push() after close() returns false and never enqueues.
//  - Items enqueued before close() remain drainable in FIFO order.
//  - pop() returns PopStatus::Item while draining, then PopStatus::Closed.
//
// A producer waits while the queue is at capacity; capacity 0 is rejected in
// the constructor with std::invalid_argument.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity)
    {
        if (capacity_ == 0) {
            throw std::invalid_argument(
                "BoundedQueue capacity must be greater than 0");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    BoundedQueue(BoundedQueue&&) = delete;
    BoundedQueue& operator=(BoundedQueue&&) = delete;

    // Test/instrumentation hook invoked just before a producer blocks on a
    // full queue. It runs while the queue mutex is held, so it must not call
    // back into this queue or block.
    void setProducerWaitCallback(std::function<void()> callback)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        producer_wait_callback_ = std::move(callback);
    }

    // As above, but for consumers about to block on an empty queue.
    void setConsumerWaitCallback(std::function<void()> callback)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        consumer_wait_callback_ = std::move(callback);
    }

    // Blocks while full. Returns false if the queue was closed (either before
    // the call or while waiting for space); a rejected item is not enqueued.
    bool push(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) {
            return false;
        }
        if (queue_.size() >= capacity_) {
            ++stats_.producer_wait_count;
            if (producer_wait_callback_) {
                producer_wait_callback_();
            }
            not_full_.wait(lock, [this] {
                return closed_ || queue_.size() < capacity_;
            });
            if (closed_) {
                return false;
            }
        }

        queue_.push_back(std::move(value));
        ++stats_.enqueued;
        if (queue_.size() > stats_.max_depth) {
            stats_.max_depth = queue_.size();
        }
        not_empty_.notify_one();
        return true;
    }

    // Blocks while empty and open. Drains items already present after close();
    // once empty and closed, returns PopStatus::Closed.
    PopStatus pop(T& value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            if (closed_) {
                return PopStatus::Closed;
            }
            ++stats_.consumer_wait_count;
            if (consumer_wait_callback_) {
                consumer_wait_callback_();
            }
            not_empty_.wait(lock, [this] {
                return closed_ || !queue_.empty();
            });
        }
        if (queue_.empty()) {
            return PopStatus::Closed;
        }

        value = std::move(queue_.front());
        queue_.pop_front();
        ++stats_.dequeued;
        not_full_.notify_one();
        return PopStatus::Item;
    }

    // Idempotent. No new items are accepted afterwards; queued items remain
    // available until drained.
    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::size_t capacity() const
    {
        return capacity_;
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool isClosed() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    QueueStats stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    bool closed_ = false;
    QueueStats stats_;
    std::function<void()> producer_wait_callback_;
    std::function<void()> consumer_wait_callback_;
};

}  // namespace edge_ai
