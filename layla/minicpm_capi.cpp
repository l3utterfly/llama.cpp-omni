#include "minicpm_capi.h"

#include "MBMtmd.h"

#include <nlohmann/json.hpp>

#include <TargetConditionals.h>
#include <mach/mach.h>
#include <os/proc.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>

struct minicpm_engine {
    mb_mtmd_context * ctx;
    minicpm_load_info info;
    std::string last_error;
};

namespace {

// The memory policy below is a port of MiniCPMV47Const and MBDeviceMemoryProbe from the
// MiniCPM-V iOS app (MiniCPM-V-Apps, MiniCPM-V-demo/Sources), which is where the
// numbers were measured. In the app it lives in Swift; Layla has no Swift layer, so it
// runs here, next to the bridge it configures.

constexpr int64_t kMiB = 1024 * 1024;
constexpr int64_t kGiB = 1024 * kMiB;

// MiniCPM-V 4.7 routes 8 experts per token per layer: a decode cache narrower than
// that cannot hold a single step.
constexpr int32_t kExpertsUsed = 8;
constexpr int32_t kDecodeWindowStep = 8;
constexpr int32_t kPrewarmLevels[] = {8, 16, 32, 48, 64, 96};
// The app pins the K4-total-1 cluster table from MBPrewarmTable.h; with a cluster
// table the bridge truncates its 96 rows per layer to the decode window.
constexpr int32_t kPrewarmTableId = 1;
// The bridge reads these as "-1 = model default"; the app's V4.7 values differ.
constexpr int32_t kImageMaxTokens = 512;
constexpr int32_t kImageMaxSliceNums = 9;

// Device tiers use thresholds below the nominal RAM because devices report less than
// they are sold with (a 16 GB iPad reports 15.14 GiB, an 8 GB iPhone 7.45 GiB).
enum class device_tier { large, medium, small, tiny };

device_tier tier_for(double gib) {
    if (gib >= 13.5) return device_tier::large;
    if (gib >= 9.5) return device_tier::medium;
    if (gib >= 6.5) return device_tier::small;
    return device_tier::tiny;
}

double device_memory_gib() {
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) != 0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / static_cast<double>(kGiB);
}

// What the OS will still let this process allocate before jetsam.
int64_t available_process_bytes() {
    return static_cast<int64_t>(os_proc_available_memory());
}

// phys_footprint is the number jetsam judges, GPU-shared anonymous pages included.
int64_t footprint_bytes() {
    task_vm_info_data_t info = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return static_cast<int64_t>(info.phys_footprint);
}

int32_t default_decode_window(device_tier tier) {
    switch (tier) {
        case device_tier::large: return 128;
        case device_tier::medium: return 48;
        case device_tier::small: return 24;
        case device_tier::tiny: return kExpertsUsed;
    }
    return kExpertsUsed;
}

int32_t max_decode_window(device_tier tier) {
    switch (tier) {
        case device_tier::large: return 144;
        case device_tier::medium: return 80;
        case device_tier::small: return 32;
        case device_tier::tiny: return 2 * kExpertsUsed;
    }
    return 2 * kExpertsUsed;
}

// 0 stays 0 (window off); anything else snaps down to the step, then up to one step's
// experts, then to the device's ceiling.
int32_t clamp_decode_window(int32_t requested, device_tier tier) {
    if (requested < kDecodeWindowStep / 2) {
        return 0;
    }
    const int32_t stepped = (requested / kDecodeWindowStep) * kDecodeWindowStep;
    return std::min(std::max(stepped, kExpertsUsed), max_decode_window(tier));
}

// The app's whole memory allowance per tier: the probe over-reports headroom, so the
// budget is also bounded by this ceiling minus what the process already holds.
int64_t memory_ceiling_bytes(device_tier tier) {
    switch (tier) {
        case device_tier::large: return 11 * kGiB;
        case device_tier::medium: return 6 * kGiB;
        case device_tier::small:
        case device_tier::tiny: return 5 * kGiB;
    }
    return 5 * kGiB;
}

// Resident memory the window must leave room for, measured by the app: baseline,
// mmproj, Metal compute, recurrent state and a 2k-token KV come to 2400 MiB. Only 10 of
// the 40 layers carry KV, at 20 KiB per token, so a longer context adds its share past
// 2k. A DSpark draft model with its context adds about 600 MiB.
int64_t reserve_bytes(int32_t n_ctx, bool spec) {
    const int64_t kv_mib = std::llround(static_cast<double>(n_ctx) * 20480.0 / static_cast<double>(kMiB));
    const int64_t kv_extra_mib = std::max<int64_t>(0, kv_mib - 40);
    return (2400 + kv_extra_mib + (spec ? 600 : 0)) * kMiB;
}

struct experts_layout {
    int64_t bytes_per_slot;
    int32_t layers;
};

// One slot holds one expert of every routed family. Each family's size is averaged over
// the layers because the quantisation differs per layer (V4.7's down projection is
// Q6_K in half the layers and Q4_K in the rest); reading layer 0 alone overestimates a
// slot by ~15% and undersizes the window.
bool read_experts_layout(const std::string & experts_dir, experts_layout & out, std::string & error) {
    const std::string manifest_path = experts_dir + "/manifest.json";
    std::ifstream in(manifest_path);
    if (!in) {
        error = "cannot open " + manifest_path;
        return false;
    }
    const nlohmann::json root = nlohmann::json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (root.is_discarded() || !root.contains("entries") || !root["entries"].is_array()) {
        error = "malformed experts manifest " + manifest_path;
        return false;
    }

    static const std::set<std::string> routed = {
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"};
    std::map<std::string, std::pair<int64_t, int64_t>> per_family;  // family -> (sum, count)
    std::set<int64_t> layers;
    for (const auto & entry : root["entries"]) {
        if (!entry.is_object()) {
            continue;
        }
        const auto family = entry.find("tensor_family");
        const auto bytes = entry.find("bytes_per_expert");
        const auto layer = entry.find("layer");
        if (family == entry.end() || !family->is_string() || routed.count(family->get<std::string>()) == 0) {
            continue;
        }
        if (bytes == entry.end() || !bytes->is_number_integer() || bytes->get<int64_t>() <= 0) {
            continue;
        }
        auto & acc = per_family[family->get<std::string>()];
        acc.first += bytes->get<int64_t>();
        acc.second += 1;
        if (layer != entry.end() && layer->is_number_integer()) {
            layers.insert(layer->get<int64_t>());
        }
    }

    int64_t total = 0;
    for (const auto & [family, acc] : per_family) {
        total += acc.first / acc.second;
    }
    if (total <= 0 || layers.empty()) {
        error = "experts manifest lists no routed experts: " + manifest_path;
        return false;
    }
    out.bytes_per_slot = total;
    out.layers = static_cast<int32_t>(layers.size());
    return true;
}

// The largest cache width c whose window (c cache slots plus prefetch_slots per layer)
// fits the memory left after the reserve. 0 when nothing fits.
int32_t budget_decode_window(const experts_layout & layout,
                             int32_t prefetch_slots,
                             int32_t n_ctx,
                             bool spec,
                             device_tier tier) {
    const int64_t available = available_process_bytes();
    const int64_t by_ceiling = std::max<int64_t>(0, memory_ceiling_bytes(tier) - footprint_bytes());
    const int64_t headroom = available > 0 ? std::min(available, by_ceiling) : by_ceiling;
    const int64_t budget = headroom - reserve_bytes(n_ctx, spec);
    if (budget <= 0) {
        return 0;
    }
    const int64_t total_slots = budget / (layout.bytes_per_slot * layout.layers);
    const int64_t cache_slots = total_slots - prefetch_slots;
    return cache_slots > 0 ? static_cast<int32_t>(std::min<int64_t>(cache_slots, INT32_MAX)) : 0;
}

int32_t resolve_decode_window(int32_t requested_raw,
                              const experts_layout & layout,
                              int32_t prefetch_slots,
                              int32_t n_ctx,
                              bool spec,
                              device_tier tier) {
    const int32_t requested = requested_raw == MINICPM_DECODE_WINDOW_AUTO
        ? default_decode_window(tier)
        : clamp_decode_window(requested_raw, tier);
    if (requested == 0) {
        return 0;
    }
    const int32_t budget = budget_decode_window(layout, prefetch_slots, n_ctx, spec, tier);
    if (budget > 0) {
        return clamp_decode_window(std::min(requested, budget), tier);
    }
    // Without a usable probe the request cannot be trusted as is; fall back to the tier.
    return std::min(requested, default_decode_window(tier));
}

// Prewarm fills the cache part of the window, so it never exceeds c.
int32_t resolve_prewarm_level(int32_t requested, int32_t decode_window) {
    if (requested <= 0 || decode_window <= 0) {
        return 0;
    }
    const int32_t cap = std::min(requested, decode_window);
    int32_t level = 0;
    for (const int32_t candidate : kPrewarmLevels) {
        if (candidate <= cap) {
            level = candidate;
        }
    }
    return level;
}

// The verify batch is n_max + 1 rows and must fit the window (slots / experts per row);
// a batch that falls out of the window measured 20 -> 2.3 tok/s in the app.
int32_t resolve_spec_n_max(int32_t requested, int32_t window_slots) {
    if (requested <= 0) {
        return 0;
    }
    const int32_t max_rows = std::max(1, window_slots / kExpertsUsed);
    return std::min(requested, std::max(0, max_rows - 1));
}

bool empty(const char * s) {
    return s == nullptr || s[0] == '\0';
}

void write_error(char * error, size_t error_size, const std::string & message) {
    if (error != nullptr && error_size > 0) {
        std::snprintf(error, error_size, "%s", message.c_str());
    }
}

}  // namespace

extern "C" {

minicpm_load_params minicpm_load_params_default(void) {
    minicpm_load_params p = {};
    p.model_path = "";
    p.mmproj_path = "";
    p.experts_path = "";
    p.draft_model_path = "";
    p.n_ctx = 4096;
    p.n_threads = 4;
    p.temperature = 0.7f;
    p.image_max_tokens = kImageMaxTokens;
    p.image_max_slice_nums = kImageMaxSliceNums;
    p.moe_slot_bank = 256;
    p.moe_decode_window = MINICPM_DECODE_WINDOW_AUTO;
    p.moe_lookahead_prefetch_depth = 8;
    p.moe_prewarm_level = 32;
    p.spec_draft_n_max = 0;
    return p;
}

minicpm_engine * minicpm_load(const minicpm_load_params * params, char * error, size_t error_size) {
    if (params == nullptr) {
        write_error(error, error_size, "minicpm_load: no parameters");
        return nullptr;
    }
    if (empty(params->model_path) || empty(params->mmproj_path) || empty(params->experts_path)) {
        write_error(error, error_size, "minicpm_load: model, mmproj and experts paths are required");
        return nullptr;
    }

    experts_layout layout = {};
    std::string layout_error;
    if (!read_experts_layout(params->experts_path, layout, layout_error)) {
        write_error(error, error_size, "minicpm_load: " + layout_error);
        return nullptr;
    }

    const double gib = device_memory_gib();
    const device_tier tier = tier_for(gib);
    const int32_t n_ctx = params->n_ctx > 0 ? params->n_ctx : 4096;
    const int32_t prefetch = std::clamp(params->moe_lookahead_prefetch_depth, 0, 16);
    const int32_t spec_requested =
        empty(params->draft_model_path) ? 0 : std::clamp(params->spec_draft_n_max, 0, 16);
    const int32_t window = resolve_decode_window(
        params->moe_decode_window, layout, prefetch, n_ctx, spec_requested > 0, tier);
    const int32_t prewarm = resolve_prewarm_level(params->moe_prewarm_level, window);
    const int32_t spec = resolve_spec_n_max(spec_requested, window + prefetch);

    mb_mtmd_params mp = mb_mtmd_params_default();
    mp.n_ctx = n_ctx;
    mp.n_threads = params->n_threads > 0 ? params->n_threads : 4;
    mp.temperature = params->temperature;
    // The simulator's Metal device aborts on the 1+ GiB mmproj buffer.
    mp.use_gpu = !TARGET_OS_SIMULATOR;
    mp.mmproj_use_gpu = !TARGET_OS_SIMULATOR;
    // The first prefill warms the graph anyway; the app skips the warmup for V4.7.
    mp.warmup = false;
    mp.image_max_tokens = params->image_max_tokens;
    mp.image_max_slice_nums = params->image_max_slice_nums;
    mp.moe_slot_bank = std::max(params->moe_slot_bank, kExpertsUsed);
    mp.moe_decode_window = window;
    mp.moe_lookahead_prefetch_depth = prefetch;
    mp.moe_prewarm_level = prewarm;
    mp.moe_prewarm_table_id = kPrewarmTableId;
    mp.spec_draft_n_max = spec;

    mb_mtmd_context * ctx = mb_mtmd_init(params->model_path,
                                         params->mmproj_path,
                                         params->experts_path,
                                         spec > 0 ? params->draft_model_path : "",
                                         &mp);
    if (ctx == nullptr) {
        write_error(error, error_size, std::string("minicpm_load: ") + mb_mtmd_last_init_error());
        return nullptr;
    }
    // An experts path already selects V4.7 inside the bridge; set it explicitly so the
    // ChatML template does not depend on that side effect. V4.7 runs with thinking off.
    mb_mtmd_set_model_version(ctx, 47);
    mb_mtmd_set_enable_thinking(ctx, false);

    auto * engine = new minicpm_engine{};
    engine->ctx = ctx;
    engine->info.moe_decode_window = window;
    engine->info.moe_lookahead_prefetch_depth = prefetch;
    engine->info.moe_prewarm_level = prewarm;
    engine->info.spec_draft_n_max = spec;
    engine->info.moe_slot_bank = mp.moe_slot_bank;
    engine->info.moe_bytes_per_slot = layout.bytes_per_slot;
    engine->info.moe_layer_count = layout.layers;
    engine->info.device_memory_gib = gib;
    return engine;
}

void minicpm_free(minicpm_engine * engine) {
    if (engine == nullptr) {
        return;
    }
    mb_mtmd_free(engine->ctx);
    delete engine;
}

minicpm_load_info minicpm_get_load_info(const minicpm_engine * engine) {
    return engine != nullptr ? engine->info : minicpm_load_info{};
}

bool minicpm_reset(minicpm_engine * engine) {
    if (engine == nullptr) {
        return false;
    }
    engine->last_error.clear();
    if (!mb_mtmd_clean_kv_cache(engine->ctx)) {
        engine->last_error = "failed to clear the KV cache";
        return false;
    }
    return true;
}

bool minicpm_add_image(minicpm_engine * engine, const char * image_path) {
    if (engine == nullptr) {
        return false;
    }
    engine->last_error.clear();
    if (mb_mtmd_prefill_image(engine->ctx, image_path) != 0) {
        engine->last_error = mb_mtmd_get_last_error(engine->ctx);
        return false;
    }
    // Kept apart from the encode in the bridge so the app could time them separately;
    // both belong to "the user attached an image", before the message is sent.
    if (mb_mtmd_prewarm_after_image(engine->ctx) != 0) {
        engine->last_error = mb_mtmd_get_last_error(engine->ctx);
        return false;
    }
    return true;
}

bool minicpm_add_text(minicpm_engine * engine, const char * role, const char * text) {
    if (engine == nullptr) {
        return false;
    }
    engine->last_error.clear();
    if (mb_mtmd_prefill_text(engine->ctx, text, role) != 0) {
        engine->last_error = mb_mtmd_get_last_error(engine->ctx);
        return false;
    }
    return true;
}

minicpm_finish_reason minicpm_generate(minicpm_engine * engine,
                                       int32_t n_predict,
                                       minicpm_token_callback on_token,
                                       void * user_data,
                                       minicpm_perf * perf) {
    if (engine == nullptr) {
        return MINICPM_FINISH_ERROR;
    }
    engine->last_error.clear();

    // mb_mtmd_loop reports a failed decode as the end of generation and records the
    // reason on the context, so a changed error string is how the two are told apart.
    const std::string error_before = mb_mtmd_get_last_error(engine->ctx);
    minicpm_finish_reason reason = MINICPM_FINISH_LENGTH;
    int32_t generated = 0;
    while (n_predict <= 0 || generated < n_predict) {
        const mb_mtmd_token token = mb_mtmd_loop(engine->ctx);
        bool keep_going = true;
        // A NULL piece mid-turn is a UTF-8 sequence still being assembled.
        if (token.token != nullptr) {
            if (on_token != nullptr && token.token[0] != '\0') {
                keep_going = on_token(token.token, user_data);
            }
            mb_mtmd_string_free(token.token);
        }
        if (token.is_end) {
            const std::string error_after = mb_mtmd_get_last_error(engine->ctx);
            if (error_after != error_before) {
                engine->last_error = error_after;
                reason = MINICPM_FINISH_ERROR;
            } else {
                reason = MINICPM_FINISH_EOS;
            }
            break;
        }
        ++generated;
        if (!keep_going) {
            reason = MINICPM_FINISH_STOPPED;
            break;
        }
    }

    // Closing the turn also parks the expert window's memory until the next decode.
    const mb_mtmd_perf p = mb_mtmd_perf_snapshot(engine->ctx);
    if (perf != nullptr) {
        perf->prefill_tokens = p.prefill_tokens;
        perf->decode_tokens = p.decode_tokens;
        perf->prefill_ms = p.prefill_ms;
        perf->decode_ms = p.decode_ms;
        perf->prep_ms = p.prep_ms;
        perf->image_tokens = p.image_tokens;
        perf->image_encode_ms = p.image_encode_ms;
    }
    return reason;
}

void minicpm_set_image_max_slice_nums(minicpm_engine * engine, int32_t n) {
    if (engine != nullptr) {
        mb_mtmd_set_image_max_slice_nums(engine->ctx, n);
    }
}

const char * minicpm_last_error(const minicpm_engine * engine) {
    return engine != nullptr ? engine->last_error.c_str() : "";
}

}  // extern "C"
