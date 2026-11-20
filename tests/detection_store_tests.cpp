#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <sqlite3.h>
#include <unistd.h>

#include "detection_store.hpp"

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

bool near(float first, float second)
{
    return std::fabs(first - second) < 1e-5f;
}

class TempDb {
public:
    TempDb()
    {
        static std::atomic<int> counter{0};
        path_ = (std::filesystem::temp_directory_path() /
                 ("edge_ai_storage_" + std::to_string(::getpid()) + "_" +
                  std::to_string(counter.fetch_add(1)) + ".db"))
                    .string();
        removeFiles();
    }

    ~TempDb() { removeFiles(); }

    const std::string& path() const { return path_; }

    void removeFiles() const
    {
        for (const char* suffix : {"", "-journal", "-wal", "-shm"}) {
            std::error_code error;
            std::filesystem::remove(path_ + suffix, error);
        }
    }

private:
    std::string path_;
};

edge_ai::DefectRecord makeDefect(int class_id,
                                 const std::string& class_name,
                                 float confidence,
                                 float x1,
                                 float y1,
                                 float x2,
                                 float y2)
{
    edge_ai::DefectRecord defect;
    defect.class_id = class_id;
    defect.class_name = class_name;
    defect.confidence = confidence;
    defect.x1 = x1;
    defect.y1 = y1;
    defect.x2 = x2;
    defect.y2 = y2;
    return defect;
}

edge_ai::DetectionEvent makeEvent(std::int64_t occurred_at_ms,
                                  const std::string& source,
                                  const std::string& product_id,
                                  bool is_ng)
{
    edge_ai::DetectionEvent event;
    event.occurred_at_ms = occurred_at_ms;
    event.source = source;
    event.product_id = product_id;
    event.is_ng = is_ng;
    event.model_identifier = "sha256-test-model";
    event.model_path = "/models/test.onnx";
    event.confidence_threshold = 0.25f;
    return event;
}

bool testRoundtrip(std::string& error)
{
    TempDb database;
    edge_ai::DetectionStore store;
    if (!expect(store.open(database.path(), true, error),
                "cannot create store: " + error, error)) {
        return false;
    }

    edge_ai::DetectionEvent ng = makeEvent(1000, "/images/ng.jpg", "P1", true);
    ng.defects.push_back(makeDefect(0, "crazing", 0.9f, 10.0f, 20.0f,
                                    110.0f, 120.0f));
    ng.defects.push_back(makeDefect(5, "scratches", 0.7f, 30.0f, 40.0f,
                                    90.0f, 80.0f));
    if (!expect(store.insertEvent(ng, error),
                "NG event insert failed: " + error, error) ||
        !expect(!ng.event_uid.empty(), "event uid was not generated", error) ||
        !expect(ng.defect_count == 2, "NG defect count is wrong", error)) {
        return false;
    }

    edge_ai::DetectionEvent ok = makeEvent(2000, "/images/ok.jpg", "P2", false);
    if (!expect(store.insertEvent(ok, error),
                "OK event insert failed: " + error, error) ||
        !expect(ok.defect_count == 0, "OK defect count is wrong", error) ||
        !expect(ok.event_uid != ng.event_uid,
                "event uids are not unique", error)) {
        return false;
    }

    std::vector<edge_ai::DetectionEvent> events;
    if (!expect(store.query({}, events, error),
                "query all failed: " + error, error) ||
        !expect(events.size() == 2, "query all did not return two events",
                error)) {
        return false;
    }
    const edge_ai::DetectionEvent& first = events[0];
    const edge_ai::DetectionEvent& second = events[1];
    if (!expect(first.occurred_at_ms == 1000 && first.is_ng &&
                    first.defect_count == 2 && first.defects.size() == 2,
                "first event fields are wrong", error) ||
        !expect(first.source == "/images/ng.jpg" &&
                    first.product_id == "P1" &&
                    first.model_identifier == "sha256-test-model",
                "first event metadata is wrong", error) ||
        !expect(first.defects[0].class_name == "crazing" &&
                    near(first.defects[0].confidence, 0.9f) &&
                    near(first.defects[0].x1, 10.0f) &&
                    near(first.defects[0].y2, 120.0f),
                "first defect fields are wrong", error) ||
        !expect(second.occurred_at_ms == 2000 && !second.is_ng &&
                    second.defects.empty(),
                "second event fields are wrong", error)) {
        return false;
    }

    edge_ai::EventQuery ng_query;
    ng_query.status = edge_ai::EventQuery::StatusFilter::Ng;
    edge_ai::EventQuery ok_query;
    ok_query.status = edge_ai::EventQuery::StatusFilter::Ok;
    edge_ai::EventQuery product_query;
    product_query.has_product_id = true;
    product_query.product_id = "P1";
    edge_ai::EventQuery from_query;
    from_query.has_from = true;
    from_query.from_ms = 1500;
    edge_ai::EventQuery to_query;
    to_query.has_to = true;
    to_query.to_ms = 1500;
    edge_ai::EventQuery combined;
    combined.has_to = true;
    combined.to_ms = 1500;
    combined.status = edge_ai::EventQuery::StatusFilter::Ng;
    combined.has_product_id = true;
    combined.product_id = "P1";

    const auto count = [&](const edge_ai::EventQuery& query,
                           std::size_t expected,
                           const std::string& label) {
        std::vector<edge_ai::DetectionEvent> result;
        std::string query_error;
        if (!store.query(query, result, query_error)) {
            error = label + " failed: " + query_error;
            return false;
        }
        if (result.size() != expected) {
            error = label + " returned " + std::to_string(result.size()) +
                    " events, expected " + std::to_string(expected);
            return false;
        }
        return true;
    };
    if (!count(ng_query, 1, "status=NG filter") ||
        !count(ok_query, 1, "status=OK filter") ||
        !count(product_query, 1, "product filter") ||
        !count(from_query, 1, "from filter") ||
        !count(to_query, 1, "to filter") ||
        !count(combined, 1, "combined filter")) {
        return false;
    }

    store.close();
    if (!expect(store.open(database.path(), false, error),
                "cannot reopen store read-only: " + error, error)) {
        return false;
    }
    std::vector<edge_ai::DetectionEvent> reopened;
    if (!expect(store.query({}, reopened, error),
                "query after reopen failed: " + error, error) ||
        !expect(reopened.size() == 2,
                "reopened store lost events", error)) {
        return false;
    }

    edge_ai::DetectionStore missing;
    std::string missing_error;
    return expect(!missing.open(database.path() + ".missing", false,
                                missing_error),
                  "read-only open of a missing database succeeded", error);
}

bool testRollbackOnDefectFailure(std::string& error)
{
    TempDb database;
    {
        edge_ai::DetectionStore store;
        if (!expect(store.open(database.path(), true, error),
                    "cannot create store: " + error, error)) {
            return false;
        }
    }

    sqlite3* db = nullptr;
    if (sqlite3_open(database.path().c_str(), &db) != SQLITE_OK) {
        error = std::string("cannot open raw database: ") +
                (db != nullptr ? sqlite3_errmsg(db) : "unknown");
        if (db != nullptr) {
            sqlite3_close(db);
        }
        return false;
    }
    const char* trigger =
        "CREATE TRIGGER fail_poison BEFORE INSERT ON detection_defects "
        "WHEN NEW.class_name = 'poison' "
        "BEGIN SELECT RAISE(ABORT, 'injected defect failure'); END;";
    char* trigger_error = nullptr;
    const int rc = sqlite3_exec(db, trigger, nullptr, nullptr, &trigger_error);
    if (rc != SQLITE_OK) {
        error = std::string("cannot create trigger: ") +
                (trigger_error != nullptr ? trigger_error
                                          : sqlite3_errmsg(db));
        sqlite3_free(trigger_error);
        sqlite3_close(db);
        return false;
    }
    sqlite3_close(db);

    edge_ai::DetectionStore store;
    if (!expect(store.open(database.path(), true, error),
                "cannot reopen store: " + error, error)) {
        return false;
    }
    edge_ai::DetectionEvent event =
        makeEvent(3000, "/images/poison.jpg", "P3", true);
    event.defects.push_back(makeDefect(0, "crazing", 0.8f, 1.0f, 2.0f,
                                       3.0f, 4.0f));
    event.defects.push_back(makeDefect(5, "poison", 0.9f, 5.0f, 6.0f,
                                       7.0f, 8.0f));
    std::string write_error;
    if (!expect(!store.insertEvent(event, write_error),
                "defect trigger did not fail the insert", error) ||
        !expect(write_error.find("injected defect failure") !=
                    std::string::npos,
                "defect failure lacks the injected diagnostic: " +
                    write_error,
                error)) {
        return false;
    }

    std::vector<edge_ai::DetectionEvent> events;
    if (!expect(store.query({}, events, error),
                "query after rollback failed: " + error, error) ||
        !expect(events.empty(),
                "main event was not rolled back after defect failure", error)) {
        return false;
    }
    return true;
}

bool testQueryFilters(std::string& error)
{
    TempDb database;
    edge_ai::DetectionStore store;
    if (!expect(store.open(database.path(), true, error),
                "cannot create store: " + error, error)) {
        return false;
    }

    edge_ai::DetectionEvent first =
        makeEvent(1000, "/images/one.jpg", "P'1", true);
    first.defects.push_back(makeDefect(0, "crazing", 0.8f, 1.0f, 2.0f,
                                       3.0f, 4.0f));
    edge_ai::DetectionEvent second =
        makeEvent(2000, "/images/two.jpg", "P2", false);
    edge_ai::DetectionEvent third =
        makeEvent(3000, "/images/three.jpg", "P'1", false);
    for (edge_ai::DetectionEvent* event : {&first, &second, &third}) {
        if (!expect(store.insertEvent(*event, error),
                    "insert failed: " + error, error)) {
            return false;
        }
    }

    edge_ai::EventQuery injection;
    injection.has_product_id = true;
    injection.product_id = "P' OR '1'='1";
    std::vector<edge_ai::DetectionEvent> result;
    if (!expect(store.query(injection, result, error),
                "injection-shaped product query failed: " + error, error) ||
        !expect(result.empty(),
                "parameter binding treated a product id as SQL", error)) {
        return false;
    }

    edge_ai::EventQuery product;
    product.has_product_id = true;
    product.product_id = "P'1";
    if (!expect(store.query(product, result, error),
                "quoted product query failed: " + error, error) ||
        !expect(result.size() == 2,
                "quoted product filter returned the wrong count", error)) {
        return false;
    }

    edge_ai::EventQuery status_ng;
    status_ng.status = edge_ai::EventQuery::StatusFilter::Ng;
    edge_ai::EventQuery status_ok;
    status_ok.status = edge_ai::EventQuery::StatusFilter::Ok;
    edge_ai::EventQuery range;
    range.has_from = true;
    range.from_ms = 1500;
    range.has_to = true;
    range.to_ms = 2500;
    if (!expect(store.query(status_ng, result, error) && result.size() == 1,
                "NG filter count is wrong", error) ||
        !expect(store.query(status_ok, result, error) && result.size() == 2,
                "OK filter count is wrong", error) ||
        !expect(store.query(range, result, error) && result.size() == 1 &&
                    result[0].occurred_at_ms == 2000,
                "time range filter is wrong", error)) {
        return false;
    }

    edge_ai::EventQuery none;
    none.has_from = true;
    none.from_ms = 4000;
    return expect(store.query(none, result, error) && result.empty(),
                  "empty query result was not handled", error);
}

}  // namespace

int main(int argc, char* argv[])
{
    const TestCase tests[] = {
        {"storage_roundtrip", testRoundtrip},
        {"storage_rollback_on_defect_failure", testRollbackOnDefectFailure},
        {"storage_query_filters", testQueryFilters},
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
        std::cerr << "error: unknown storage test: " << filter << "\n";
        return 2;
    }
    std::cout << ran << " test(s), " << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
