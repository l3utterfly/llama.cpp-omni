#pragma once

// Layla's C ABI over llama.cpp-omni for MiniCPM-V 4.7 35B-A3B with MoE-SSD expert
// streaming. The React Native bridge includes only this header; everything behind it
// (llama.cpp-omni, its ggml, the vendored MiniCPM-V-Apps bridge) lives inside the
// minicpm framework image, so none of it meets llama-cpp's different ggml revision.
//
// One engine owns one conversation in its KV cache. Calls on an engine must not
// overlap; the bridge serialises them on its worker thread. The engine configures
// llama.cpp-omni through process-wide environment variables (LLAMA_MOE_*), so only one
// engine may be loaded at a time.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct minicpm_engine minicpm_engine;

// minicpm_load_params.moe_decode_window: size the per-layer expert cache from the
// device's memory, as the MiniCPM-V iOS app does on first launch.
#define MINICPM_DECODE_WINDOW_AUTO (-1)

typedef struct minicpm_load_params {
    // The expert-stripped main file, e.g. MiniCPM-V-4_7-35B-A3B-mix_q3gud_im-les.gguf.
    const char * model_path;
    // The vision tower (mmproj-model-f16.gguf). Required: the engine always runs the
    // multimodal path, which is the one the MiniCPM-V app ships and tunes.
    const char * mmproj_path;
    // The experts/ directory holding manifest.json and the layer_NNN.bin shards that
    // were split from the same model as model_path.
    const char * experts_path;
    // DSpark draft model for speculative decoding. "" disables speculative decoding.
    const char * draft_model_path;

    int32_t n_ctx;
    int32_t n_threads;
    // Sampling temperature; 0 = greedy. The sampler is built at load, so changing it
    // needs a reload.
    float temperature;
    // The physical batch is not a parameter: the bridge sizes it itself, starting small
    // and growing it for the first long or image prompt (capped at 1024 while the
    // decode window is active).

    // Token budget per image and the llava-uhd slice cap; -1 = model default.
    int32_t image_max_tokens;
    int32_t image_max_slice_nums;

    // Expert slots in the shared bank prefill installs into, one layer at a time.
    int32_t moe_slot_bank;
    // Per-layer decode cache width in expert slots ("c" in the MiniCPM-V app).
    // MINICPM_DECODE_WINDOW_AUTO picks the device default; 0 turns the window off;
    // a positive value is a request that is still clamped to the memory budget.
    int32_t moe_decode_window;
    // Next-layer experts predicted and prefetched per step; 0 = off. The prefetch
    // slots are added to the decode window, outside the c cache slots.
    int32_t moe_lookahead_prefetch_depth;
    // Experts per layer read into the decode window before the first token; 0 = off.
    // Capped to the decode window that was actually granted.
    int32_t moe_prewarm_level;
    // DSpark draft tokens per round; 0 = off. Clamped so the verify batch still fits
    // the decode window, and forced to 0 without a draft model.
    int32_t spec_draft_n_max;
} minicpm_load_params;

// The values the MiniCPM-V iOS app ships for MiniCPM-V 4.7 35B-A3B, with every path "".
minicpm_load_params minicpm_load_params_default(void);

// What minicpm_load settled on after applying the memory policy.
typedef struct minicpm_load_info {
    int32_t moe_decode_window;
    int32_t moe_lookahead_prefetch_depth;
    int32_t moe_prewarm_level;
    int32_t spec_draft_n_max;
    int32_t moe_slot_bank;
    // Bytes of one expert slot across the routed families, averaged over layers, and
    // the number of routed layers, both read from the experts manifest.
    int64_t moe_bytes_per_slot;
    int32_t moe_layer_count;
    double device_memory_gib;
} minicpm_load_info;

// Loads the model, the vision tower and the expert streamer. Paths are borrowed for
// the duration of the call. Returns NULL on failure and writes the reason into error.
minicpm_engine * minicpm_load(const minicpm_load_params * params, char * error, size_t error_size);

void minicpm_free(minicpm_engine * engine);

minicpm_load_info minicpm_get_load_info(const minicpm_engine * engine);

// Clears the KV cache so the next turn starts a new conversation.
bool minicpm_reset(minicpm_engine * engine);

// Encodes an image now and folds it into the next user turn. The vision encode and the
// expert prewarm run here, so call it as soon as the user attaches the image rather
// than when the message is sent. Returns false on failure; see minicpm_last_error.
bool minicpm_add_image(minicpm_engine * engine, const char * image_path);

// Appends one ChatML turn to the KV cache. role is "system", "user" or "assistant".
// A user turn ends with the assistant header (thinking off), so minicpm_generate can
// follow it directly. An assistant turn is history the model did not generate here.
bool minicpm_add_text(minicpm_engine * engine, const char * role, const char * text);

// Receives each complete UTF-8 piece. Return false to stop generating.
typedef bool (*minicpm_token_callback)(const char * piece, void * user_data);

typedef enum minicpm_finish_reason {
    // The model ended its turn; the KV cache holds a closed assistant turn and the
    // next minicpm_add_text continues the conversation.
    MINICPM_FINISH_EOS = 0,
    // n_predict was reached, or the callback returned false. The assistant turn is
    // still open in the KV cache: reset and replay the conversation before the next
    // turn.
    MINICPM_FINISH_LENGTH = 1,
    MINICPM_FINISH_STOPPED = 2,
    MINICPM_FINISH_ERROR = 3,
} minicpm_finish_reason;

typedef struct minicpm_perf {
    int32_t prefill_tokens;
    int32_t decode_tokens;
    double prefill_ms;
    double decode_ms;
    // Send-turn work outside prefill: vision reload, image decode, tokenisation.
    double prep_ms;
    // The eager image encode done in minicpm_add_image.
    int32_t image_tokens;
    double image_encode_ms;
} minicpm_perf;

// Generates the assistant reply to the turns added since the last generation.
// n_predict <= 0 means no limit beyond the context. When the turn ends, the expert
// window's memory is released until the next decode, as the MiniCPM-V app does between
// turns; perf (may be NULL) receives the turn's counters.
minicpm_finish_reason minicpm_generate(minicpm_engine * engine,
                                       int32_t n_predict,
                                       minicpm_token_callback on_token,
                                       void * user_data,
                                       minicpm_perf * perf);

// Overrides the slice cap for the next image; -1 restores the model default.
void minicpm_set_image_max_slice_nums(minicpm_engine * engine, int32_t n);

// The last error on this engine, "" when there is none. Valid until the next call.
const char * minicpm_last_error(const minicpm_engine * engine);

#ifdef __cplusplus
}
#endif
