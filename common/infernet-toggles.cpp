#include "infernet-toggles.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>

double infernet_toggle(const char * name, double def) {
    static std::mutex mu;
    static std::map<std::string, double> vals;
    static long long mtime_seen = -1;
    static const char * path = [] { const char * p = getenv("INFERNET_TOGGLES"); return p ? p : "/tmp/infernet-toggles.json"; }();

    std::lock_guard<std::mutex> lk(mu);

    struct stat st;
    const long long mtime = stat(path, &st) == 0 ? (long long) st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec : 0;
    if (mtime != mtime_seen) {
        mtime_seen = mtime;
        vals.clear();
        if (mtime != 0) {
            try {
                std::ifstream f(path);
                const auto j = nlohmann::json::parse(f);
                for (auto it = j.begin(); it != j.end(); ++it) {
                    if (it.value().is_number() || it.value().is_boolean()) {
                        vals[it.key()] = it.value().is_boolean() ? (it.value().get<bool>() ? 1.0 : 0.0) : it.value().get<double>();
                    }
                }
            } catch (...) {
                // a half-written file: keep the env/defaults until the next change
            }
        }
    }

    const auto it = vals.find(name);
    if (it != vals.end()) {
        return it->second;
    }
    const char * e = getenv(name);
    return e ? atof(e) : def;
}

FILE * infernet_capture_file(const char * suffix) {
    static const char * prefix = getenv("LLAMA_SELECTOR_CAPTURE");
    if (!prefix) {
        return nullptr;
    }
    static std::mutex mu;
    static std::map<std::string, FILE *> files;
    std::lock_guard<std::mutex> lk(mu);
    auto & f = files[suffix];
    if (!f) {
        f = fopen((std::string(prefix) + suffix).c_str(), "ab");
    }
    return f;
}

int64_t & infernet_capture_round() {
    static int64_t r = 0;
    return r;
}
