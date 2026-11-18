#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bounded_queue.hpp"

using namespace std::chrono_literals;

namespace {

using TestFunction = bool (*)(std::string&);

struct TestCase {
    const char* name;
    TestFunction function;
};

bool expect(bool condition, const std::string& message, std::string& error)
{
    if (!condition) {
        error = message;
        return false;
    }
    return true;
}

template <typename Predicate>
bool waitUntil(Predicate predicate, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return predicate();
}

bool testRejectZeroCapacity(std::string& error)
{
    try {
        edge_ai::BoundedQueue<int> queue(0);
        (void)queue;
        return expect(false, "capacity 0 was accepted", error);
    } catch (const std::invalid_argument& exception) {
        return expect(std::string(exception.what()).find("capacity") !=
                          std::string::npos,
                      "capacity-0 diagnostic is unclear: " +
                          std::string(exception.what()),
                      error);
    } catch (const std::exception& exception) {
        return expect(false,
                      "capacity 0 threw the wrong exception: " +
                          std::string(exception.what()),
                      error);
    }
}

bool testFullBlocks(std::string& error)
{
    edge_ai::BoundedQueue<int> queue(2);
    if (!expect(queue.push(1), "failed to push A", error) ||
        !expect(queue.push(2), "failed to push B", error)) {
        return false;
    }

    std::promise<bool> push_result;
    std::future<bool> pushed = push_result.get_future();
    std::thread producer([&] { push_result.set_value(queue.push(3)); });

    if (!waitUntil(
            [&] { return queue.stats().producer_wait_count >= 1; }, 2000ms)) {
        queue.close();
        producer.join();
        return expect(false, "producer did not wait on a full queue", error);
    }
    if (!expect(pushed.wait_for(100ms) == std::future_status::timeout,
                "C enqueued while the queue was full", error)) {
        queue.close();
        producer.join();
        return false;
    }

    const edge_ai::QueueStats full_stats = queue.stats();
    if (!expect(full_stats.enqueued == 2,
                "queue accepted C before A was consumed", error) ||
        !expect(full_stats.max_depth == 2,
                "max depth is not 2 while full", error)) {
        queue.close();
        producer.join();
        return false;
    }

    int value = 0;
    if (!expect(queue.pop(value) == edge_ai::PopStatus::Item && value == 1,
                "first pop is not A", error)) {
        queue.close();
        producer.join();
        return false;
    }

    if (pushed.wait_for(2000ms) != std::future_status::ready) {
        queue.close();
        producer.join();
        return expect(false, "C was not accepted after A was consumed", error);
    }
    if (!expect(pushed.get(), "C push was rejected", error)) {
        producer.join();
        return false;
    }

    if (!expect(queue.pop(value) == edge_ai::PopStatus::Item && value == 2,
                "second pop is not B", error) ||
        !expect(queue.pop(value) == edge_ai::PopStatus::Item && value == 3,
                "third pop is not C", error)) {
        producer.join();
        return false;
    }
    producer.join();

    const edge_ai::QueueStats stats = queue.stats();
    if (!expect(stats.enqueued == 3 && stats.dequeued == 3,
                "full-load enqueue/dequeue counts are wrong", error) ||
        !expect(stats.max_depth == 2, "full-load max depth is not 2", error) ||
        !expect(stats.producer_wait_count >= 1,
                "full-load producer wait was not recorded", error)) {
        return false;
    }

    std::cout << "full-block report: capacity=2 enqueued=" << stats.enqueued
              << " dequeued=" << stats.dequeued
              << " max_depth=" << stats.max_depth
              << " producer_wait_count=" << stats.producer_wait_count
              << " consumer_wait_count=" << stats.consumer_wait_count << "\n";
    return true;
}

bool testEmptyBlocks(std::string& error)
{
    edge_ai::BoundedQueue<int> queue(2);
    std::atomic<int> received{-1};
    std::promise<edge_ai::PopStatus> pop_result;
    std::future<edge_ai::PopStatus> popped = pop_result.get_future();
    std::thread consumer([&] {
        int value = 0;
        const edge_ai::PopStatus status = queue.pop(value);
        if (status == edge_ai::PopStatus::Item) {
            received.store(value);
        }
        pop_result.set_value(status);
    });

    if (!waitUntil(
            [&] { return queue.stats().consumer_wait_count >= 1; }, 2000ms)) {
        queue.close();
        consumer.join();
        return expect(false, "consumer did not wait on an empty queue", error);
    }
    if (!expect(queue.push(42), "failed to wake the empty consumer", error)) {
        queue.close();
        consumer.join();
        return false;
    }
    if (popped.wait_for(2000ms) != std::future_status::ready) {
        queue.close();
        consumer.join();
        return expect(false, "consumer was not woken by a push", error);
    }
    const edge_ai::PopStatus status = popped.get();
    consumer.join();
    if (!expect(status == edge_ai::PopStatus::Item,
                "consumer did not receive an item", error) ||
        !expect(received.load() == 42, "consumer received the wrong item",
                error)) {
        return false;
    }
    const edge_ai::QueueStats stats = queue.stats();
    return expect(stats.enqueued == 1 && stats.dequeued == 1,
                  "empty-queue enqueue/dequeue counts are wrong", error) &&
           expect(stats.consumer_wait_count >= 1,
                  "empty-queue consumer wait was not recorded", error);
}

bool testCloseDrains(std::string& error)
{
    edge_ai::BoundedQueue<int> queue(2);
    if (!expect(queue.push(10) && queue.push(20),
                "failed to preload close test", error)) {
        return false;
    }
    queue.close();
    queue.close();

    if (!expect(queue.isClosed(), "queue did not report closed", error) ||
        !expect(!queue.push(30), "push succeeded after close", error)) {
        return false;
    }

    int value = 0;
    if (!expect(queue.pop(value) == edge_ai::PopStatus::Item && value == 10,
                "close did not preserve first queued item", error) ||
        !expect(queue.pop(value) == edge_ai::PopStatus::Item && value == 20,
                "close did not preserve second queued item", error) ||
        !expect(queue.pop(value) == edge_ai::PopStatus::Closed,
                "closed queue did not return Closed after draining", error) ||
        !expect(queue.pop(value) == edge_ai::PopStatus::Closed,
                "Closed status is not repeatable", error)) {
        return false;
    }

    const edge_ai::QueueStats stats = queue.stats();
    return expect(stats.enqueued == 2 && stats.dequeued == 2,
                  "close-drain counts are wrong", error);
}

bool testCloseWakesWaiters(std::string& error)
{
    {
        edge_ai::BoundedQueue<int> queue(1);
        if (!expect(queue.push(1), "failed to fill producer-wake queue", error)) {
            return false;
        }
        std::promise<bool> result;
        std::future<bool> future = result.get_future();
        std::thread producer([&] { result.set_value(queue.push(2)); });
        if (!waitUntil(
                [&] { return queue.stats().producer_wait_count >= 1; },
                2000ms)) {
            queue.close();
            producer.join();
            return expect(false, "producer did not block before close", error);
        }
        queue.close();
        if (future.wait_for(2000ms) != std::future_status::ready) {
            producer.join();
            return expect(false, "close did not wake the producer", error);
        }
        const bool accepted = future.get();
        producer.join();
        if (!expect(!accepted, "producer enqueued after close", error)) {
            return false;
        }
    }

    {
        edge_ai::BoundedQueue<int> queue(1);
        std::promise<edge_ai::PopStatus> result;
        std::future<edge_ai::PopStatus> future = result.get_future();
        std::thread consumer([&] {
            int value = 0;
            result.set_value(queue.pop(value));
        });
        if (!waitUntil(
                [&] { return queue.stats().consumer_wait_count >= 1; },
                2000ms)) {
            queue.close();
            consumer.join();
            return expect(false, "consumer did not block before close", error);
        }
        queue.close();
        if (future.wait_for(2000ms) != std::future_status::ready) {
            consumer.join();
            return expect(false, "close did not wake the consumer", error);
        }
        const edge_ai::PopStatus status = future.get();
        consumer.join();
        if (!expect(status == edge_ai::PopStatus::Closed,
                    "woken consumer did not get Closed", error)) {
            return false;
        }
    }
    return true;
}

bool testConcurrentNoLoss(std::string& error)
{
    constexpr std::size_t kCapacity = 4;
    constexpr int kCount = 2000;
    edge_ai::BoundedQueue<int> queue(kCapacity);
    std::vector<int> received;
    received.reserve(kCount);
    bool push_failed = false;

    std::thread consumer([&] {
        int value = 0;
        while (queue.pop(value) == edge_ai::PopStatus::Item) {
            received.push_back(value);
            std::this_thread::sleep_for(50us);
        }
    });
    std::thread producer([&] {
        for (int i = 0; i < kCount; ++i) {
            if (!queue.push(i)) {
                push_failed = true;
                break;
            }
        }
        queue.close();
    });

    producer.join();
    consumer.join();

    if (!expect(!push_failed, "producer push was rejected", error) ||
        !expect(received.size() == static_cast<std::size_t>(kCount),
                "consumer received the wrong number of items", error)) {
        return false;
    }
    for (int i = 0; i < kCount; ++i) {
        if (!expect(received[static_cast<std::size_t>(i)] == i,
                    "sequence has a missing, duplicate or reordered item",
                    error)) {
            return false;
        }
    }

    const edge_ai::QueueStats stats = queue.stats();
    if (!expect(stats.enqueued == static_cast<std::size_t>(kCount) &&
                    stats.dequeued == static_cast<std::size_t>(kCount),
                "concurrent enqueue/dequeue counts are wrong", error) ||
        !expect(stats.max_depth <= kCapacity && stats.max_depth > 0,
                "max depth is outside [1, capacity]", error) ||
        !expect(stats.producer_wait_count >= 1,
                "slow consumer did not cause producer waiting", error)) {
        return false;
    }

    std::cout << "concurrency report: capacity=" << kCapacity
              << " enqueued=" << stats.enqueued
              << " dequeued=" << stats.dequeued
              << " max_depth=" << stats.max_depth
              << " producer_wait_count=" << stats.producer_wait_count
              << " consumer_wait_count=" << stats.consumer_wait_count << "\n";
    return true;
}

}  // namespace

int main(int argc, char* argv[])
{
    const TestCase tests[] = {
        {"queue_reject_zero_capacity", testRejectZeroCapacity},
        {"queue_full_blocks", testFullBlocks},
        {"queue_empty_blocks", testEmptyBlocks},
        {"queue_close_drains", testCloseDrains},
        {"queue_close_wakes_waiters", testCloseWakesWaiters},
        {"queue_concurrent_no_loss", testConcurrentNoLoss},
    };

    const std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0;
    int failures = 0;
    for (const TestCase& test : tests) {
        if (!filter.empty() && filter != test.name) {
            continue;
        }
        ++ran;
        std::string error;
        if (test.function(error)) {
            std::cout << "[PASS] " << test.name << "\n";
        } else {
            ++failures;
            std::cout << "[FAIL] " << test.name << ": " << error << "\n";
        }
    }
    if (ran == 0) {
        std::cerr << "error: unknown queue test: " << filter << "\n";
        return 2;
    }
    std::cout << ran << " test(s), " << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
