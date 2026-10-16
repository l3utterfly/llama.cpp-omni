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

// Load progress, as llama-server reports it: stage is "text_model" (the main file) or
// "mmproj_model" (the vision tower), progress runs from 0 to 1, or is negative while a
// stage's progress cannot be measured. Called on the loading thread, often; the caller
// throttles.
typedef void (*minicpm_load_progress_callback)(const char * stage, float progress, void * user_data);

typedef struct minicpm_load_params {
    // The expert-stripped main file, e.g. MiniCPM-V-4_7-35B-A3B-mix_q3gud_im-les.gguf.
    const char * model_path;
    // The vision tower (mmproj-model-f16.gguf). Required: the engine always runs the
    // multimodal path, which is the one the MiniCPM-V app ships and tunes.
    const char * mmproj_path;
    // The experts/ directory holding manifest.json and the layer_NNN.bin shards that
    // were split from the same model as model_path.
    const char * experts_path;

    int32_t n_ctx;
    int32_t n_threads;
    // Sampling is per request (minicpm_chat_completions). The physical batch is not a parameter: the bridge sizes it itself, starting small
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
    // Not offered: the MiniCPM-V app's expert prewarm and DSpark speculative decoding.
    // Both hang off the app's own turn-by-turn flow (prewarm runs between picking an
    // image and sending it; DSpark replaces the sampling loop), which requests that
    // carry the whole conversation and sample like llama-server do not go through.

    // NULL = no progress. user_data is passed back unchanged.
    minicpm_load_progress_callback on_progress;
    void * user_data;
} minicpm_load_params;

// The values the MiniCPM-V iOS app ships for MiniCPM-V 4.7 35B-A3B, with every path "".
minicpm_load_params minicpm_load_params_default(void);

// What minicpm_load settled on after applying the memory policy.
typedef struct minicpm_load_info {
    int32_t moe_decode_window;
    int32_t moe_lookahead_prefetch_depth;
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

// ---- OpenAI Chat Completions (minicpm_oai.cpp) ---------------------------------
//
// Requests are llama-server /v1/chat/completions bodies and are read by llama-server's
// own code: the prompt comes from the model's chat template in its GGUF (tools, tool
// results, reasoning, images, continue_final_message), sampling, stop and limits from
// the body, with grammars and lazy tool-call grammars, and the reply is parsed into
// content, reasoning_content and tool_calls as llama-server parses it. Images are
// image_url parts (data: URLs); audio is refused, as the model has no audio tower.
// Layla's reasoning_effort: "none" turns thinking off (chat_template_kwargs wins).
//
// The engine keeps the conversation it last finished in its KV cache and, when the
// next prompt extends it, prefills only the new part. A prompt that shares only some of
// it (a regenerated reply, an edited message) prefills from the start: most of the
// model's layers are recurrent and cannot be rolled back.

// Receives each chat.completion.chunk as JSON, on the calling thread.
typedef void (*minicpm_chunk_callback)(const char * chunk_json, void * user_data);
// Polled before each generated token; true stops the generation, which then ends as
// if the model had stopped (finish_reason "stop", partial content).
typedef bool (*minicpm_cancel_callback)(void * user_data);

// Serves one request. Streams chunks when the request sets "stream": true. Returns the
// finished chat.completion as JSON, to be released with minicpm_string_free, or NULL
// with the reason in minicpm_last_error. A prompt that does not fit the context fails
// with llama-server's error body, {"error":{"type":"exceed_context_size_error",
// "n_prompt_tokens":..,"n_ctx":..}}, so callers can trim and retry the same way.
char * minicpm_chat_completions(minicpm_engine * engine,
                                const char * request_json,
                                minicpm_chunk_callback on_chunk,
                                minicpm_cancel_callback is_cancelled,
                                void * user_data);

// The prompt text a request becomes, exactly as the engine feeds it (images appear as
// mtmd's media marker). Reads only the template and vocabulary from model_path, so it
// needs no loaded engine. Returns a string to release with minicpm_string_free, or
// NULL with the reason in error.
char * minicpm_render_prompt(const char * model_path,
                             const char * request_json,
                             char * error,
                             size_t error_size);

// What the model's chat template can express, as llama-server reports it.
typedef struct minicpm_template_caps {
    bool supports_tools;
    bool supports_tool_calls;
    bool supports_parallel_tool_calls;
    bool supports_system_role;
    bool supports_preserve_reasoning;
    bool supports_reasoning_effort;
    bool supports_string_content;
    bool supports_typed_content;
    bool supports_object_arguments;
} minicpm_template_caps;

// Reads only the template from model_path. False, with the reason in error, when the
// file cannot be read or has no chat template.
bool minicpm_read_template_caps(const char * model_path,
                                minicpm_template_caps * caps,
                                char * error,
                                size_t error_size);

// ---- Saved conversations (minicpm_oai.cpp) ----------------------------------------
//
// One file holds what the KV cache holds together with the engine's record of which
// prompt and images built it, so a conversation resumes after the engine is freed and
// loaded again: when the next request extends the restored prompt, only the new part
// is prefilled. As above, a request that does not extend it prefills from the start, so
// a file from another conversation costs a read and nothing else.

// Writes the conversation in the KV cache to path. Only a reply the model ended itself
// leaves one a request can extend; anything else (no request yet, a cancelled or failed
// one) fails, so a file saved earlier is not replaced by a state nothing could resume.
// Returns the bytes written, or -1 with the reason in minicpm_last_error.
int64_t minicpm_save_state(minicpm_engine * engine, const char * path);

// Replaces the conversation in the KV cache with the one saved in path. Returns the
// tokens restored, or -1 with the reason in minicpm_last_error when the file is missing,
// truncated or was not written by this engine for a context that fits; the cache is then
// left empty, so the next request prefills as it would have anyway.
int32_t minicpm_restore_state(minicpm_engine * engine, const char * path);

void minicpm_string_free(char * s);

// Overrides the slice cap for the next image; -1 restores the model default.
void minicpm_set_image_max_slice_nums(minicpm_engine * engine, int32_t n);

// The last error on this engine, "" when there is none. Valid until the next call.
const char * minicpm_last_error(const minicpm_engine * engine);

#ifdef __cplusplus
}
#endif
