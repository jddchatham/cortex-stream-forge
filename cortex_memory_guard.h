#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct CortexMemoryGuard {
    static long get_current_rss_kb() {
        FILE * f = fopen("/proc/self/status", "r");
        if (!f) return 0;
        char line[256];
        long rss = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line + 6, "%ld", &rss);
                break;
            }
        }
        fclose(f);
        return rss;
    }

    static void check(const char * phase) {
        const char * limit_env = std::getenv("CORTEX_RSS_LIMIT_MB");
        const char * audit_env = std::getenv("CORTEX_MEMORY_AUDIT");
        long limit_mb = limit_env ? std::atol(limit_env) : 477;
        long rss_kb = get_current_rss_kb();
        double rss_mb = rss_kb / 1024.0;

        if (audit_env && std::atoi(audit_env) != 0) {
            std::printf("[CORTEX MEMORY AUDIT] Phase '%s': VmRSS = %.2f MiB (Limit: %ld MiB)\n", phase, rss_mb, limit_mb);
        }

        if (rss_mb > limit_mb) {
            std::fprintf(stderr, "[CORTEX MEMORY GUARD] CRITICAL: Hard memory ceiling exceeded at phase '%s'! VmRSS = %.2f MiB > %ld MiB limit. Aborting to protect system stability.\n",
                phase, rss_mb, limit_mb);
            std::exit(137);
        }
    }
};
