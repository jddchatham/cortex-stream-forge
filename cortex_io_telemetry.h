#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>

struct StepTelemetry {
    int step_index = 0;
    uint64_t layer_bytes_requested = 0;
    double layer_read_ms = 0.0;
    double effective_bw_mb_s = 0.0;
    double compute_ms = 0.0;
    double lm_head_read_ms = 0.0;
    double lm_head_compute_ms = 0.0;
    double total_decode_ms = 0.0;
    long vm_rss_kb = 0;
    long rss_anon_kb = 0;
    long rss_file_kb = 0;
    long vm_swap_kb = 0;
    long minor_faults = 0;
    long major_faults = 0;
    uint64_t rchar = 0;
    uint64_t wchar = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    std::string io_state = "UNKNOWN";
    std::string active_policy = "NONE";
    std::string transition_reason = "NONE";
};

class CortexIOTelemetry {
public:
    enum LogMode { DISABLED = 0, SUMMARY = 1, DIAGNOSTIC = 2 };

    static void get_process_memory(long & rss_kb, long & anon_kb, long & file_kb, long & swap_kb) {
        rss_kb = anon_kb = file_kb = swap_kb = 0;
        FILE * f = fopen("/proc/self/status", "r");
        if (!f) return;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line + 6, "%ld", &rss_kb);
            else if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line + 8, "%ld", &anon_kb);
            else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line + 8, "%ld", &file_kb);
            else if (strncmp(line, "VmSwap:", 7) == 0) sscanf(line + 7, "%ld", &swap_kb);
        }
        fclose(f);
    }

    static void get_page_faults(long & min_flt, long & maj_flt) {
        min_flt = maj_flt = 0;
        FILE * f = fopen("/proc/self/stat", "r");
        if (!f) return;
        // Fields: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt
        char comm[256];
        int pid, ppid, pgrp, session, tty_nr, tpgid;
        char state;
        unsigned long flags, cminflt, cmajflt;
        unsigned long u_minflt = 0, u_majflt = 0;
        if (fscanf(f, "%d %s %c %d %d %d %d %d %lu %lu %lu %lu %lu",
                   &pid, comm, &state, &ppid, &pgrp, &session, &tty_nr, &tpgid,
                   &flags, &u_minflt, &cminflt, &u_majflt, &cmajflt) == 13) {
            min_flt = (long)u_minflt;
            maj_flt = (long)u_majflt;
        }
        fclose(f);
    }

    static void get_io_stats(uint64_t & rchar, uint64_t & wchar, uint64_t & read_bytes, uint64_t & write_bytes) {
        rchar = wchar = read_bytes = write_bytes = 0;
        FILE * f = fopen("/proc/self/io", "r");
        if (!f) return;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "rchar:", 6) == 0) sscanf(line + 6, "%lu", &rchar);
            else if (strncmp(line, "wchar:", 6) == 0) sscanf(line + 6, "%lu", &wchar);
            else if (strncmp(line, "read_bytes:", 11) == 0) sscanf(line + 11, "%lu", &read_bytes);
            else if (strncmp(line, "write_bytes:", 12) == 0) sscanf(line + 12, "%lu", &write_bytes);
        }
        fclose(f);
    }
};
