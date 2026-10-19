#pragma once
#include <string>
#include <deque>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

enum class IOState {
    UNKNOWN,
    WARM,       // Reads fast (> 2500 MB/s), low physical storage IO, 0 major faults
    MIXED,      // Intermediate bandwidth (1200 - 2500 MB/s), intermittent physical IO
    COLD,       // Bandwidth < 1200 MB/s, sustained physical storage read traffic
    PRESSURED   // High major page faults or severe I/O stall
};

enum class AdvicePolicy {
    NONE = 0,
    SEQUENTIAL = 1,
    WILLNEED = 2
};

struct AdaptiveDecision {
    IOState state = IOState::UNKNOWN;
    AdvicePolicy policy = AdvicePolicy::NONE;
    int fadvise_flag = -1; // -1 means do not call posix_fadvise
    std::string reason = "INIT";
    bool state_changed = false;
};

class CortexAdaptiveIOController {
public:
    CortexAdaptiveIOController(bool enabled = false) 
        : m_enabled(enabled), m_current_state(IOState::UNKNOWN), m_current_policy(AdvicePolicy::NONE) {
        if (const char * env = std::getenv("CORTEX_ADAPTIVE_IO")) {
            m_enabled = (std::atoi(env) != 0);
        }
    }

    bool is_enabled() const { return m_enabled; }

    AdaptiveDecision evaluate_step(double step_bw_mb_s, uint64_t step_phys_read_bytes, long step_maj_faults) {
        AdaptiveDecision dec;
        if (!m_enabled) {
            dec.state = IOState::UNKNOWN;
            dec.policy = AdvicePolicy::NONE;
            dec.fadvise_flag = -1;
            dec.reason = "ADAPTIVE_DISABLED";
            return dec;
        }

        m_bw_history.push_back(step_bw_mb_s);
        if (m_bw_history.size() > m_window_size) m_bw_history.pop_front();

        m_phys_io_history.push_back(step_phys_read_bytes);
        if (m_phys_io_history.size() > m_window_size) m_phys_io_history.pop_front();

        m_maj_flt_history.push_back(step_maj_faults);
        if (m_maj_flt_history.size() > m_window_size) m_maj_flt_history.pop_front();

        if (m_bw_history.size() < 2) {
            dec.state = m_current_state;
            dec.policy = m_current_policy;
            dec.fadvise_flag = get_fadvise_flag(m_current_policy);
            dec.reason = "WARMUP_WINDOW";
            return dec;
        }

        double avg_bw = 0.0;
        for (double b : m_bw_history) avg_bw += b;
        avg_bw /= m_bw_history.size();

        uint64_t total_phys_read = 0;
        for (uint64_t p : m_phys_io_history) total_phys_read += p;

        long total_maj_flt = 0;
        for (long f : m_maj_flt_history) total_maj_flt += f;

        IOState candidate_state = m_current_state;

        if (total_maj_flt > 5) {
            candidate_state = IOState::PRESSURED;
        } else if (avg_bw > 2500.0 && total_phys_read < 10 * 1024 * 1024) {
            candidate_state = IOState::WARM;
        } else if (avg_bw < 1200.0 || total_phys_read > 500 * 1024 * 1024) {
            candidate_state = IOState::COLD;
        } else {
            candidate_state = IOState::MIXED;
        }

        if (candidate_state != m_current_state) {
            if (candidate_state == m_pending_state) {
                m_pending_count++;
                if (m_pending_count >= m_hysteresis_limit) {
                    IOState old_state = m_current_state;
                    m_current_state = candidate_state;
                    m_current_policy = select_policy_for_state(m_current_state);
                    dec.state_changed = true;
                    dec.reason = std::string("TRANSITION_") + state_name(old_state) + "_TO_" + state_name(m_current_state);
                    m_pending_count = 0;
                } else {
                    dec.reason = std::string("HYSTERESIS_PENDING (") + std::to_string(m_pending_count) + "/" + std::to_string(m_hysteresis_limit) + ")";
                }
            } else {
                m_pending_state = candidate_state;
                m_pending_count = 1;
                dec.reason = std::string("HYSTERESIS_INIT (") + state_name(candidate_state) + ")";
            }
        } else {
            m_pending_count = 0;
            dec.reason = std::string("STATE_STABLE (") + state_name(m_current_state) + ")";
        }

        dec.state = m_current_state;
        dec.policy = m_current_policy;
        dec.fadvise_flag = get_fadvise_flag(m_current_policy);
        return dec;
    }

    static const char * state_name(IOState s) {
        switch (s) {
            case IOState::WARM: return "WARM";
            case IOState::MIXED: return "MIXED";
            case IOState::COLD: return "COLD";
            case IOState::PRESSURED: return "PRESSURED";
            default: return "UNKNOWN";
        }
    }

    static const char * policy_name(AdvicePolicy p) {
        switch (p) {
            case AdvicePolicy::NONE: return "NONE";
            case AdvicePolicy::SEQUENTIAL: return "SEQUENTIAL";
            case AdvicePolicy::WILLNEED: return "WILLNEED";
            default: return "UNKNOWN";
        }
    }

private:
    static AdvicePolicy select_policy_for_state(IOState state) {
        switch (state) {
            case IOState::COLD:
                return AdvicePolicy::WILLNEED;
            case IOState::MIXED:
                return AdvicePolicy::SEQUENTIAL;
            case IOState::WARM:
                return AdvicePolicy::NONE;
            case IOState::PRESSURED:
                return AdvicePolicy::SEQUENTIAL;
            default:
                return AdvicePolicy::NONE;
        }
    }

    static int get_fadvise_flag(AdvicePolicy policy) {
        switch (policy) {
            case AdvicePolicy::SEQUENTIAL: return POSIX_FADV_SEQUENTIAL;
            case AdvicePolicy::WILLNEED: return POSIX_FADV_WILLNEED;
            default: return -1;
        }
    }

    bool m_enabled = false;
    IOState m_current_state = IOState::UNKNOWN;
    AdvicePolicy m_current_policy = AdvicePolicy::NONE;

    IOState m_pending_state = IOState::UNKNOWN;
    int m_pending_count = 0;
    const int m_hysteresis_limit = 2;
    const size_t m_window_size = 4;

    std::deque<double> m_bw_history;
    std::deque<uint64_t> m_phys_io_history;
    std::deque<long> m_maj_flt_history;
};
