#include "detection_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

#include <sqlite3.h>
#include <unistd.h>

namespace edge_ai {
namespace {

constexpr const char* kSupportedSchemaVersion = "1";

constexpr const char* kSchemaSql = R"sql(
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');

CREATE TABLE IF NOT EXISTS detection_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid            TEXT NOT NULL UNIQUE,
    occurred_at_ms       INTEGER NOT NULL,
    source               TEXT NOT NULL,
    product_id           TEXT,
    is_ng                INTEGER NOT NULL CHECK (is_ng IN (0, 1)),
    defect_count         INTEGER NOT NULL CHECK (defect_count >= 0),
    model_identifier     TEXT NOT NULL,
    model_path           TEXT,
    image_path           TEXT,
    confidence_threshold REAL NOT NULL,
    created_at_ms        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS detection_defects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    INTEGER NOT NULL REFERENCES detection_events(id) ON DELETE CASCADE,
    class_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    x1          REAL NOT NULL,
    y1          REAL NOT NULL,
    x2          REAL NOT NULL,
    y2          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_occurred
    ON detection_events(occurred_at_ms);
CREATE INDEX IF NOT EXISTS idx_events_status
    ON detection_events(is_ng);
CREATE INDEX IF NOT EXISTS idx_events_product
    ON detection_events(product_id);
CREATE INDEX IF NOT EXISTS idx_defects_event
    ON detection_defects(event_id);
)sql";

std::string sqliteMessage(sqlite3* db, const std::string& context)
{
    const char* message = db != nullptr ? sqlite3_errmsg(db)
                                        : "unknown SQLite error";
    return context + ": " + message;
}

bool execSql(sqlite3* db,
             const char* sql,
             std::string& error,
             const std::string& context)
{
    char* sqlite_error = nullptr;
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &sqlite_error);
    if (rc != SQLITE_OK) {
        error = context + ": " +
                (sqlite_error != nullptr ? sqlite_error
                                         : sqlite3_errmsg(db));
        sqlite3_free(sqlite_error);
        return false;
    }
    return true;
}

class Statement {
public:
    Statement() = default;
    ~Statement()
    {
        if (stmt_ != nullptr) {
            sqlite3_finalize(stmt_);
        }
    }

    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    bool prepare(sqlite3* db, const char* sql, std::string& error,
                 const std::string& context)
    {
        reset();
        const int rc = sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr);
        if (rc != SQLITE_OK) {
            error = sqliteMessage(db, context + " (prepare)");
            return false;
        }
        return true;
    }

    sqlite3_stmt* get() const { return stmt_; }

    bool resetAndClear(std::string& error, const std::string& context)
    {
        if (stmt_ == nullptr) {
            error = context + ": statement is not prepared";
            return false;
        }
        const int rc = sqlite3_reset(stmt_);
        sqlite3_clear_bindings(stmt_);
        if (rc != SQLITE_OK) {
            error = sqliteMessage(sqlite3_db_handle(stmt_),
                                  context + " (reset)");
            return false;
        }
        return true;
    }

private:
    void reset()
    {
        if (stmt_ != nullptr) {
            sqlite3_finalize(stmt_);
            stmt_ = nullptr;
        }
    }

    sqlite3_stmt* stmt_ = nullptr;
};

class Transaction {
public:
    explicit Transaction(sqlite3* db) : db_(db) {}

    ~Transaction()
    {
        if (active_) {
            sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        }
    }

    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;

    bool begin(std::string& error)
    {
        return execSql(db_, "BEGIN IMMEDIATE;", error,
                       "database transaction start failed");
    }

    bool commit(std::string& error)
    {
        if (!execSql(db_, "COMMIT;", error,
                     "database transaction commit failed")) {
            return false;
        }
        active_ = false;
        return true;
    }

private:
    sqlite3* db_ = nullptr;
    bool active_ = true;
};

bool bindText(sqlite3_stmt* stmt,
              int index,
              const std::string& value,
              std::string& error)
{
    if (sqlite3_bind_text(stmt, index, value.c_str(), -1,
                          SQLITE_TRANSIENT) != SQLITE_OK) {
        error = sqliteMessage(sqlite3_db_handle(stmt),
                              "database text bind failed");
        return false;
    }
    return true;
}

bool bindNullableText(sqlite3_stmt* stmt,
                      int index,
                      const std::string& value,
                      std::string& error)
{
    if (value.empty()) {
        if (sqlite3_bind_null(stmt, index) != SQLITE_OK) {
            error = sqliteMessage(sqlite3_db_handle(stmt),
                                  "database null bind failed");
            return false;
        }
        return true;
    }
    return bindText(stmt, index, value, error);
}

bool bindInt(sqlite3_stmt* stmt, int index, int value, std::string& error)
{
    if (sqlite3_bind_int(stmt, index, value) != SQLITE_OK) {
        error = sqliteMessage(sqlite3_db_handle(stmt),
                              "database integer bind failed");
        return false;
    }
    return true;
}

bool bindInt64(sqlite3_stmt* stmt,
               int index,
               std::int64_t value,
               std::string& error)
{
    if (sqlite3_bind_int64(stmt, index, static_cast<sqlite3_int64>(value)) !=
        SQLITE_OK) {
        error = sqliteMessage(sqlite3_db_handle(stmt),
                              "database integer bind failed");
        return false;
    }
    return true;
}

bool bindDouble(sqlite3_stmt* stmt,
                int index,
                double value,
                std::string& error)
{
    if (sqlite3_bind_double(stmt, index, value) != SQLITE_OK) {
        error = sqliteMessage(sqlite3_db_handle(stmt),
                              "database real bind failed");
        return false;
    }
    return true;
}

bool stepDone(sqlite3_stmt* stmt, std::string& error,
              const std::string& context)
{
    const int rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        error = sqliteMessage(sqlite3_db_handle(stmt), context);
        return false;
    }
    return true;
}

std::string columnText(sqlite3_stmt* stmt, int column)
{
    const unsigned char* text = sqlite3_column_text(stmt, column);
    return text != nullptr ? reinterpret_cast<const char*>(text) : "";
}

// Reads schema_meta existence and version without creating anything.
bool schemaMetaStatus(sqlite3* db,
                      bool& exists,
                      std::string& version,
                      std::string& error)
{
    exists = false;
    version.clear();
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(
            db,
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'schema_meta' LIMIT 1;",
            -1, &stmt, nullptr) != SQLITE_OK) {
        error = std::string("database schema probe failed: ") +
                sqlite3_errmsg(db);
        return false;
    }
    int rc = sqlite3_step(stmt);
    const bool has_table = rc == SQLITE_ROW;
    sqlite3_finalize(stmt);
    if (rc != SQLITE_ROW && rc != SQLITE_DONE) {
        error = std::string("database schema probe failed: ") +
                sqlite3_errmsg(db);
        return false;
    }
    exists = has_table;
    if (!has_table) {
        return true;
    }

    stmt = nullptr;
    if (sqlite3_prepare_v2(
            db, "SELECT value FROM schema_meta WHERE key = 'schema_version';",
            -1, &stmt, nullptr) != SQLITE_OK) {
        error = std::string("database schema metadata is unreadable: ") +
                sqlite3_errmsg(db);
        return false;
    }
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        const unsigned char* text = sqlite3_column_text(stmt, 0);
        if (text != nullptr) {
            version = reinterpret_cast<const char*>(text);
        }
    }
    sqlite3_finalize(stmt);
    if (rc != SQLITE_ROW) {
        error = "database schema_version row is missing";
        return false;
    }
    return true;
}

bool verifySchema(sqlite3* db, const std::string& path, std::string& error)
{
    const char* checks[] = {
        "SELECT 1 FROM detection_events LIMIT 0;",
        "SELECT 1 FROM detection_defects LIMIT 0;",
    };
    for (const char* sql : checks) {
        sqlite3_stmt* stmt = nullptr;
        const int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
        sqlite3_finalize(stmt);
        if (rc != SQLITE_OK) {
            error = "database schema not found in '" + path + "': " +
                    sqlite3_errmsg(db);
            return false;
        }
    }

    // Reject an unsupported schema version before any write can proceed.
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(
            db, "SELECT value FROM schema_meta WHERE key = 'schema_version';",
            -1, &stmt, nullptr) != SQLITE_OK) {
        error = "database schema metadata is missing in '" + path + "': " +
                sqlite3_errmsg(db);
        return false;
    }
    const int rc = sqlite3_step(stmt);
    std::string version;
    if (rc == SQLITE_ROW) {
        const unsigned char* text = sqlite3_column_text(stmt, 0);
        if (text != nullptr) {
            version = reinterpret_cast<const char*>(text);
        }
    }
    sqlite3_finalize(stmt);
    if (version != kSupportedSchemaVersion) {
        error = "unsupported database schema version '" + version +
                "' in '" + path + "' (supported: " +
                kSupportedSchemaVersion + ")";
        return false;
    }
    return true;
}

}  // namespace

struct DetectionStore::Impl {
    sqlite3* db = nullptr;
};

DetectionStore::DetectionStore() : impl_(std::make_unique<Impl>()) {}

DetectionStore::~DetectionStore()
{
    close();
}

bool DetectionStore::open(const std::string& path,
                          bool create,
                          std::string& error)
{
    error.clear();
    close();
    if (path.empty()) {
        error = "database path is empty";
        return false;
    }

    // Only persistent file databases are supported. Reject SQLite's in-memory
    // and URI forms so a "saved" event cannot silently disappear when the
    // process exits.
    if (path == ":memory:" || path.rfind("file:", 0) == 0) {
        error = "database path must be a persistent file, got '" + path + "'";
        return false;
    }

    if (create) {
        const std::filesystem::path parent =
            std::filesystem::path(path).parent_path();
        if (!parent.empty()) {
            std::error_code fs_error;
            std::filesystem::create_directories(parent, fs_error);
            if (fs_error) {
                error = "cannot create database directory '" +
                        parent.string() + "': " + fs_error.message();
                return false;
            }
        }
    }

    sqlite3* db = nullptr;
    const int base_flags = create ? (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
                                  : SQLITE_OPEN_READONLY;
    const int rc =
        sqlite3_open_v2(path.c_str(), &db, base_flags | SQLITE_OPEN_FULLMUTEX,
                        nullptr);
    if (rc != SQLITE_OK) {
        const std::string message =
            db != nullptr ? sqlite3_errmsg(db) : sqlite3_errstr(rc);
        error = "cannot open database '" + path + "': " + message;
        if (db != nullptr) {
            sqlite3_close_v2(db);
        }
        return false;
    }

    impl_->db = db;
    sqlite3_busy_timeout(db, 5000);
    if (!execSql(db, "PRAGMA foreign_keys = ON;", error,
                 "database pragma failed")) {
        close();
        return false;
    }
    // Check the existing schema version BEFORE any DDL or write. Unknown
    // versions must be rejected without touching the database.
    bool meta_exists = false;
    std::string version;
    if (!schemaMetaStatus(db, meta_exists, version, error)) {
        close();
        return false;
    }
    if (meta_exists) {
        if (version != kSupportedSchemaVersion) {
            error = "unsupported database schema version '" + version +
                    "' in '" + path + "' (supported: " +
                    kSupportedSchemaVersion + ")";
            close();
            return false;
        }
    } else {
        if (!create) {
            error = "database schema not found in '" + path + "'";
            close();
            return false;
        }
        bool initialized = false;
        {
            // Initialize a new database atomically: tables, indices and the
            // schema_version row either all appear or none do.
            Transaction transaction(db);
            initialized =
                transaction.begin(error) &&
                execSql(db, kSchemaSql, error,
                        "database schema creation failed") &&
                transaction.commit(error);
            // On any failure the Transaction destructor rolls back.
        }
        if (!initialized) {
            close();
            return false;
        }
    }
    // Verify the tables and the (now known supported) version once more.
    if (!verifySchema(db, path, error)) {
        close();
        return false;
    }
    return true;
}

void DetectionStore::close() noexcept
{
    if (impl_ && impl_->db != nullptr) {
        sqlite3_close_v2(impl_->db);
        impl_->db = nullptr;
    }
}

bool DetectionStore::isOpen() const
{
    return impl_ && impl_->db != nullptr;
}

bool DetectionStore::insertEvent(DetectionEvent& event, std::string& error)
{
    error.clear();
    if (!isOpen()) {
        error = "database is not open";
        return false;
    }
    if (event.source.empty()) {
        error = "event source is empty";
        return false;
    }
    if (event.model_identifier.empty()) {
        error = "event model identifier is empty";
        return false;
    }

    sqlite3* db = impl_->db;
    event.defect_count = event.defects.size();
    if (event.event_uid.empty()) {
        event.event_uid = generateEventUid();
    }
    if (event.occurred_at_ms == 0) {
        event.occurred_at_ms =
            toEpochMillis(std::chrono::system_clock::now());
    }
    const std::int64_t created_at_ms =
        toEpochMillis(std::chrono::system_clock::now());

    Transaction transaction(db);
    if (!transaction.begin(error)) {
        return false;
    }

    {
        Statement insert_event;
        const char* event_sql =
            "INSERT INTO detection_events "
            "(event_uid, occurred_at_ms, source, product_id, is_ng, "
            " defect_count, model_identifier, model_path, image_path, "
            " confidence_threshold, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
        if (!insert_event.prepare(db, event_sql, error,
                                  "database event insert failed") ||
            !bindText(insert_event.get(), 1, event.event_uid, error) ||
            !bindInt64(insert_event.get(), 2, event.occurred_at_ms, error) ||
            !bindText(insert_event.get(), 3, event.source, error) ||
            !bindNullableText(insert_event.get(), 4, event.product_id,
                              error) ||
            !bindInt(insert_event.get(), 5, event.is_ng ? 1 : 0, error) ||
            !bindInt64(insert_event.get(), 6,
                       static_cast<std::int64_t>(event.defect_count), error) ||
            !bindText(insert_event.get(), 7, event.model_identifier, error) ||
            !bindNullableText(insert_event.get(), 8, event.model_path,
                              error) ||
            !bindNullableText(insert_event.get(), 9, event.image_path,
                              error) ||
            !bindDouble(insert_event.get(), 10,
                        static_cast<double>(event.confidence_threshold),
                        error) ||
            !bindInt64(insert_event.get(), 11, created_at_ms, error) ||
            !stepDone(insert_event.get(), error,
                      "database event insert failed")) {
            return false;
        }

        const sqlite3_int64 event_id = sqlite3_last_insert_rowid(db);
        if (!event.defects.empty()) {
            Statement insert_defect;
            const char* defect_sql =
                "INSERT INTO detection_defects "
                "(event_id, class_id, class_name, confidence, "
                " x1, y1, x2, y2) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
            if (!insert_defect.prepare(db, defect_sql, error,
                                       "database defect insert failed")) {
                return false;
            }
            for (const DefectRecord& defect : event.defects) {
                if (!insert_defect.resetAndClear(
                        error, "database defect insert failed") ||
                    !bindInt64(insert_defect.get(), 1,
                               static_cast<std::int64_t>(event_id), error) ||
                    !bindInt(insert_defect.get(), 2, defect.class_id, error) ||
                    !bindText(insert_defect.get(), 3, defect.class_name,
                              error) ||
                    !bindDouble(insert_defect.get(), 4,
                                static_cast<double>(defect.confidence),
                                error) ||
                    !bindDouble(insert_defect.get(), 5,
                                static_cast<double>(defect.x1), error) ||
                    !bindDouble(insert_defect.get(), 6,
                                static_cast<double>(defect.y1), error) ||
                    !bindDouble(insert_defect.get(), 7,
                                static_cast<double>(defect.x2), error) ||
                    !bindDouble(insert_defect.get(), 8,
                                static_cast<double>(defect.y2), error) ||
                    !stepDone(insert_defect.get(), error,
                              "database defect insert failed")) {
                    return false;
                }
            }
        }
    }

    return transaction.commit(error);
}

bool DetectionStore::query(const EventQuery& query,
                           std::vector<DetectionEvent>& events,
                           std::string& error)
{
    error.clear();
    events.clear();
    if (!isOpen()) {
        error = "database is not open";
        return false;
    }

    sqlite3* db = impl_->db;
    std::string sql =
        "SELECT id, event_uid, occurred_at_ms, source, product_id, is_ng, "
        "       defect_count, model_identifier, model_path, image_path, "
        "       confidence_threshold "
        "FROM detection_events";
    std::vector<std::string> conditions;
    if (query.has_from) {
        conditions.push_back("occurred_at_ms >= ?");
    }
    if (query.has_to) {
        conditions.push_back("occurred_at_ms <= ?");
    }
    if (query.status != EventQuery::StatusFilter::Any) {
        conditions.push_back("is_ng = ?");
    }
    if (query.has_product_id) {
        conditions.push_back("product_id = ?");
    }
    if (!conditions.empty()) {
        sql += " WHERE ";
        for (std::size_t i = 0; i < conditions.size(); ++i) {
            if (i != 0) {
                sql += " AND ";
            }
            sql += conditions[i];
        }
    }
    sql += " ORDER BY occurred_at_ms ASC, id ASC;";

    Statement select_events;
    if (!select_events.prepare(db, sql.c_str(), error,
                               "database query failed")) {
        return false;
    }
    int index = 1;
    if (query.has_from &&
        !bindInt64(select_events.get(), index++, query.from_ms, error)) {
        return false;
    }
    if (query.has_to &&
        !bindInt64(select_events.get(), index++, query.to_ms, error)) {
        return false;
    }
    if (query.status != EventQuery::StatusFilter::Any &&
        !bindInt(select_events.get(), index++,
                 query.status == EventQuery::StatusFilter::Ng ? 1 : 0,
                 error)) {
        return false;
    }
    if (query.has_product_id &&
        !bindText(select_events.get(), index++, query.product_id, error)) {
        return false;
    }

    Statement select_defects;
    if (!select_defects.prepare(
            db,
            "SELECT class_id, class_name, confidence, x1, y1, x2, y2 "
            "FROM detection_defects WHERE event_id = ? ORDER BY id ASC;",
            error, "database defect query failed")) {
        return false;
    }

    int rc = SQLITE_ROW;
    while ((rc = sqlite3_step(select_events.get())) == SQLITE_ROW) {
        DetectionEvent event;
        const sqlite3_int64 event_id = sqlite3_column_int64(
            select_events.get(), 0);
        event.event_uid = columnText(select_events.get(), 1);
        event.occurred_at_ms = static_cast<std::int64_t>(
            sqlite3_column_int64(select_events.get(), 2));
        event.source = columnText(select_events.get(), 3);
        event.product_id = columnText(select_events.get(), 4);
        event.is_ng = sqlite3_column_int(select_events.get(), 5) != 0;
        event.defect_count = static_cast<std::size_t>(
            sqlite3_column_int64(select_events.get(), 6));
        event.model_identifier = columnText(select_events.get(), 7);
        event.model_path = columnText(select_events.get(), 8);
        event.image_path = columnText(select_events.get(), 9);
        event.confidence_threshold = static_cast<float>(
            sqlite3_column_double(select_events.get(), 10));

        if (!select_defects.resetAndClear(error,
                                          "database defect query failed") ||
            !bindInt64(select_defects.get(), 1, event_id, error)) {
            events.clear();
            return false;
        }
        int defect_rc = SQLITE_ROW;
        while ((defect_rc = sqlite3_step(select_defects.get())) ==
               SQLITE_ROW) {
            DefectRecord defect;
            defect.class_id = sqlite3_column_int(select_defects.get(), 0);
            defect.class_name = columnText(select_defects.get(), 1);
            defect.confidence = static_cast<float>(
                sqlite3_column_double(select_defects.get(), 2));
            defect.x1 = static_cast<float>(
                sqlite3_column_double(select_defects.get(), 3));
            defect.y1 = static_cast<float>(
                sqlite3_column_double(select_defects.get(), 4));
            defect.x2 = static_cast<float>(
                sqlite3_column_double(select_defects.get(), 5));
            defect.y2 = static_cast<float>(
                sqlite3_column_double(select_defects.get(), 6));
            event.defects.push_back(defect);
        }
        if (defect_rc != SQLITE_DONE) {
            error = sqliteMessage(db, "database defect query failed");
            events.clear();
            return false;
        }
        events.push_back(std::move(event));
    }
    if (rc != SQLITE_DONE) {
        error = sqliteMessage(db, "database query failed");
        events.clear();
        return false;
    }
    return true;
}

std::string generateEventUid()
{
    // Seed a thread-local 64-bit generator from 256 bits of random_device
    // entropy instead of a single 32-bit value. A single 32-bit seed can
    // only produce 2^32 distinct sequences and was observed to collide in
    // bulk generation. Mix in process id, a monotonic counter and a clock
    // tick so IDs stay unique within a process even if the entropy source
    // repeats, then form a UUID-v4-shaped 128-bit value.
    thread_local std::mt19937_64 generator = [] {
        std::random_device device;
        std::array<std::uint32_t, 8> seed{};
        for (std::uint32_t& word : seed) {
            word = device();
        }
        std::seed_seq sequence(seed.begin(), seed.end());
        return std::mt19937_64(sequence);
    }();
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t sequence =
        counter.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t tick = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const std::uint64_t pid = static_cast<std::uint64_t>(::getpid());
    std::uint64_t high = generator() ^ (tick + pid);
    std::uint64_t low = generator() ^ sequence ^ (tick << 1);
    high = (high & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
    low = (low & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

    char buffer[40] = {};
    std::snprintf(
        buffer, sizeof(buffer), "%08llx-%04llx-%04llx-%04llx-%012llx",
        static_cast<unsigned long long>(high >> 32),
        static_cast<unsigned long long>((high >> 16) & 0xFFFF),
        static_cast<unsigned long long>(high & 0xFFFF),
        static_cast<unsigned long long>((low >> 48) & 0xFFFF),
        static_cast<unsigned long long>(low & 0xFFFFFFFFFFFFULL));
    return buffer;
}

std::int64_t toEpochMillis(FrameTimestamp timestamp)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               timestamp.time_since_epoch())
        .count();
}

std::string formatEpochMillisIso(std::int64_t epoch_ms)
{
    std::time_t seconds = static_cast<std::time_t>(epoch_ms / 1000);
    int millis = static_cast<int>(epoch_ms % 1000);
    if (millis < 0) {
        millis += 1000;
        seconds -= 1;
    }
    std::tm utc{};
    gmtime_r(&seconds, &utc);
    char text[32] = {};
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%S", &utc);
    char result[40] = {};
    std::snprintf(result, sizeof(result), "%s.%03dZ", text, millis);
    return result;
}

bool parseTwoDigits(const std::string& text,
                    std::size_t position,
                    int& value)
{
    if (position + 2 > text.size() ||
        std::isdigit(static_cast<unsigned char>(text[position])) == 0 ||
        std::isdigit(static_cast<unsigned char>(text[position + 1])) == 0) {
        return false;
    }
    value = (text[position] - '0') * 10 + (text[position + 1] - '0');
    return true;
}

int daysInMonth(int year, int month)
{
    static const int kDays[] = {31, 28, 31, 30, 31, 30,
                                31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) {
        return 0;
    }
    if (month == 2) {
        const bool leap =
            (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        return leap ? 29 : 28;
    }
    return kDays[month - 1];
}

bool parseTimestampToEpochMillis(const std::string& text,
                                 std::int64_t& epoch_ms,
                                 std::string& error)
{
    if (text.empty()) {
        error = "timestamp is empty";
        return false;
    }

    // Integer epoch milliseconds.
    try {
        std::size_t consumed = 0;
        const long long value = std::stoll(text, &consumed);
        if (consumed == text.size()) {
            epoch_ms = static_cast<std::int64_t>(value);
            return true;
        }
    } catch (const std::exception&) {
        // Fall through to ISO-8601 parsing.
    }

    // Strict YYYY-MM-DDTHH:MM:SS[.mmm][Z|+00:00|-00:00]. std::get_time()
    // normalizes impossible calendar dates (2026-02-31 -> March), so parse
    // and validate each field explicitly.
    const std::string invalid =
        "invalid timestamp '" + text +
        "' (use epoch milliseconds or YYYY-MM-DDTHH:MM:SSZ)";
    if (text.size() < 19 || text[4] != '-' || text[7] != '-' ||
        text[10] != 'T' || text[13] != ':' || text[16] != ':') {
        error = invalid;
        return false;
    }
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    // Parse the year as four digits without the two-digit helper.
    if (text.size() < 19 ||
        std::isdigit(static_cast<unsigned char>(text[0])) == 0 ||
        std::isdigit(static_cast<unsigned char>(text[1])) == 0 ||
        std::isdigit(static_cast<unsigned char>(text[2])) == 0 ||
        std::isdigit(static_cast<unsigned char>(text[3])) == 0) {
        error = invalid;
        return false;
    }
    year = (text[0] - '0') * 1000 + (text[1] - '0') * 100 +
           (text[2] - '0') * 10 + (text[3] - '0');
    if (!parseTwoDigits(text, 5, month) ||
        !parseTwoDigits(text, 8, day) ||
        !parseTwoDigits(text, 11, hour) ||
        !parseTwoDigits(text, 14, minute) ||
        !parseTwoDigits(text, 17, second)) {
        error = invalid;
        return false;
    }
    if (month < 1 || month > 12 || day < 1 ||
        day > daysInMonth(year, month) || hour > 23 || minute > 59 ||
        second > 59) {
        error = "invalid calendar date/time in timestamp '" + text + "'";
        return false;
    }

    std::size_t position = 19;
    int millis = 0;
    if (position < text.size() && text[position] == '.') {
        ++position;
        const std::size_t start = position;
        while (position < text.size() &&
               std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
            ++position;
        }
        const std::size_t digits = position - start;
        if (digits == 0 || digits > 3) {
            error = "invalid fractional seconds in timestamp '" + text + "'";
            return false;
        }
        std::string fraction = text.substr(start, digits);
        while (fraction.size() < 3) {
            fraction.push_back('0');
        }
        millis = std::stoi(fraction);
    }
    if (position < text.size()) {
        const std::string zone = text.substr(position);
        if (zone != "Z" && zone != "z" && zone != "+00:00" &&
            zone != "-00:00") {
            error = "unsupported timezone in timestamp '" + text +
                    "' (use UTC, e.g. 2026-11-20T10:00:00Z)";
            return false;
        }
    }

    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
    const std::time_t seconds = timegm(&tm);
    epoch_ms = static_cast<std::int64_t>(seconds) * 1000 + millis;
    return true;
}

}  // namespace edge_ai
