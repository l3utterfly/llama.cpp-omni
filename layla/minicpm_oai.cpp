// OpenAI Chat Completions for MiniCPM-V; the contract is in minicpm_capi.h.
//
// Everything about the chat format is llama-server's own code, compiled in from
// tools/server: oaicompat_chat_params_parse renders the request with the model's chat
// template (tools, tool results, reasoning, images, continuation), params_from_json_cmpl
// reads sampling, stops and limits, common_sampler samples (grammars and lazy tool-call
// grammars included), and task_result_state parses the output into content, reasoning
// and tool calls, streamed as the same diffs llama-server sends. The MiniCPM-V bridge
// does the work it is tuned for: prefill and the MoE decode step.

#include "minicpm_engine.h"

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "sampling.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-task.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char * kModelName = "minicpm-v-4.7-35b-a3b";

// A prompt that does not fit, reported in llama-server's shape.
struct context_overflow : std::runtime_error {
    context_overflow(int64_t prompt_tokens, int32_t n_ctx)
        : std::runtime_error(json{
              {"error",
               {{"code", 400},
                {"message", "the request exceeds the available context size, try increasing it"},
                {"type", "exceed_context_size_error"},
                {"n_prompt_tokens", prompt_tokens},
                {"n_ctx", n_ctx}}}}
                                 .dump()) {}
};

std::string dump(const json & value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

// Layla's provider asks for "thinking off" with reasoning_effort: "none", which newer
// llama-server versions map onto the template's enable_thinking. This revision of the
// server code predates that mapping, so it is applied here; an explicit
// chat_template_kwargs.enable_thinking still wins.
void apply_reasoning_effort(json & body) {
    if (!body.contains("reasoning_effort") || !body["reasoning_effort"].is_string()) {
        return;
    }
    if (body["reasoning_effort"].get<std::string>() != "none") {
        return;
    }
    json & kwargs = body["chat_template_kwargs"];
    if (!kwargs.is_object()) {
        kwargs = json::object();
    }
    if (!kwargs.contains("enable_thinking")) {
        kwargs["enable_thinking"] = false;
    }
}

void configure_chat(server_chat_params & chat, common_chat_templates_ptr tmpls) {
    chat.use_jinja = true;
    chat.prefill_assistant = true;
    chat.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    chat.tmpls = std::move(tmpls);
    chat.allow_image = true;
    chat.allow_audio = false;
    chat.enable_thinking = true;
}

// The server marks media with a random marker so user text cannot forge one; the
// bridge's mtmd context expects mtmd's default marker.
std::string with_mtmd_marker(std::string prompt) {
    const std::string from = get_media_marker();
    const std::string to = mtmd_default_marker();
    if (from == to) {
        return prompt;
    }
    for (size_t at = prompt.find(from); at != std::string::npos; at = prompt.find(from, at + to.size())) {
        prompt.replace(at, from.size(), to);
    }
    return prompt;
}

std::string image_key(const raw_buffer & bytes) {
    const std::string_view view(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return std::to_string(std::hash<std::string_view>{}(view)) + ":" + std::to_string(bytes.size());
}

std::string completion_id() {
    static std::atomic<uint64_t> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "chatcmpl-minicpm-" + std::to_string(now) + "-" + std::to_string(counter.fetch_add(1));
}

// How much of text's end could still grow into a stop string, so it is held back.
size_t partial_stop_length(const std::string & text, const std::vector<std::string> & stops) {
    size_t longest = 0;
    for (const auto & stop : stops) {
        if (stop.empty()) {
            continue;
        }
        for (size_t n = std::min(stop.size() - 1, text.size()); n > longest; --n) {
            if (text.compare(text.size() - n, n, stop, 0, n) == 0) {
                longest = n;
                break;
            }
        }
    }
    return longest;
}

size_t find_stop(const std::string & text, const std::vector<std::string> & stops, size_t from) {
    size_t first = std::string::npos;
    for (const auto & stop : stops) {
        if (stop.empty()) {
            continue;
        }
        const size_t at = text.find(stop, from);
        if (at < first) {
            first = at;
        }
    }
    return first;
}

struct stream_context {
    minicpm_chunk_callback on_chunk;
    void * user_data;
    bool enabled;
    std::string id;
    int64_t created;

    void send(json chunk) const {
        if (!enabled || on_chunk == nullptr) {
            return;
        }
        chunk["id"] = id;
        chunk["object"] = "chat.completion.chunk";
        chunk["created"] = created;
        chunk["model"] = kModelName;
        on_chunk(dump(chunk).c_str(), user_data);
    }

    void delta(json delta) const {
        send({{"choices", json::array({{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", nullptr}}})}});
    }

    void diffs(const std::vector<common_chat_msg_diff> & diffs) const {
        for (const auto & diff : diffs) {
            delta(server_chat_msg_diff_to_json_oaicompat(diff));
        }
    }
};

// Prompt processing progress, sent as llama-server sends it for return_progress: the
// whole prompt's tokens, the ones the cache already held, and the ones in the cache so
// far, cached ones included, so the overall progress is processed / total.
struct prefill_progress {
    const stream_context * stream;
    int32_t cache_n;
    std::chrono::steady_clock::time_point started;
};

void send_prefill_progress(int32_t n_done, int32_t n_total, void * user_data) {
    const auto * p = static_cast<const prefill_progress *>(user_data);
    const auto elapsed = std::chrono::steady_clock::now() - p->started;
    p->stream->send({{"choices", json::array()},
                     {"prompt_progress",
                      {{"total", p->cache_n + n_total},
                       {"cache", p->cache_n},
                       {"processed", p->cache_n + n_done},
                       {"time_ms", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()}}}});
}

json timings_json(const mb_mtmd_perf & perf, int32_t cache_n) {
    auto per_second = [](double n, double ms) { return ms > 0 ? n * 1000.0 / ms : 0.0; };
    return {
        {"cache_n", cache_n},
        {"prompt_n", perf.prefill_tokens},
        {"prompt_ms", perf.prefill_ms},
        {"prompt_per_second", per_second(perf.prefill_tokens, perf.prefill_ms)},
        {"predicted_n", perf.decode_tokens},
        {"predicted_ms", perf.decode_ms},
        {"predicted_per_second", per_second(perf.decode_tokens, perf.decode_ms)},
    };
}

bool starts_with(const std::vector<std::string> & all, const std::vector<std::string> & prefix) {
    return prefix.size() <= all.size() && std::equal(prefix.begin(), prefix.end(), all.begin());
}

char * copy_string(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

std::string serve(minicpm_engine * e,
                  const char * request_json,
                  minicpm_chunk_callback on_chunk,
                  minicpm_cancel_callback is_cancelled,
                  void * user_data) {
    json body = json::parse(request_json != nullptr ? request_json : "");
    apply_reasoning_effort(body);

    // The request as llama-server reads it.
    std::vector<raw_buffer> images;
    const json data = oaicompat_chat_params_parse(body, e->chat, images);
    const llama_model * model = mb_mtmd_layla_model(e->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_ctx = mb_mtmd_layla_n_ctx(e->ctx);
    const common_params defaults{};  // llama-server's defaults
    task_params task = server_task::params_from_json_cmpl(vocab, defaults, n_ctx, {}, data);
    const std::string prompt = with_mtmd_marker(data.at("prompt").get<std::string>());
    std::vector<std::string> image_keys;
    for (const auto & image : images) {
        image_keys.push_back(image_key(image));
    }

    // Reuse the cache when this prompt extends it; otherwise start over. Three in four of
    // the model's layers are recurrent, and recurrent state cannot be rolled back, so a
    // prompt that shares only part of the cache (a regenerated reply, an edited message)
    // prefills from the start. The record is taken out while the request runs, so a
    // failure leaves the next request starting over rather than trusting a half-written
    // cache.
    std::string cached_prompt = std::move(e->cached_prompt);
    std::vector<std::string> cached_images = std::move(e->cached_images);
    e->cached_prompt.clear();
    e->cached_images.clear();
    const bool reuse = !cached_prompt.empty() && prompt.size() > cached_prompt.size() &&
                       prompt.compare(0, cached_prompt.size(), cached_prompt) == 0 &&
                       starts_with(image_keys, cached_images);
    size_t first_new_image = 0;
    std::string suffix = prompt;
    if (reuse) {
        suffix = prompt.substr(cached_prompt.size());
        first_new_image = cached_images.size();
    } else {
        if (!mb_mtmd_clean_kv_cache(e->ctx)) {
            throw std::runtime_error("failed to clear the KV cache");
        }
        e->kv_tokens = 0;
    }
    const int32_t cache_n = e->kv_tokens;

    std::vector<const unsigned char *> image_data;
    std::vector<size_t> image_sizes;
    for (size_t i = first_new_image; i < images.size(); ++i) {
        image_data.push_back(images[i].data());
        image_sizes.push_back(images[i].size());
    }
    const stream_context stream{on_chunk, user_data, task.stream, completion_id(),
                                static_cast<int64_t>(std::time(nullptr))};
    prefill_progress progress{&stream, cache_n, std::chrono::steady_clock::now()};
    int32_t new_tokens = 0;
    const int rc = mb_mtmd_layla_prefill_prompt(e->ctx, suffix.c_str(), image_data.data(), image_sizes.data(),
                                                static_cast<int32_t>(image_data.size()), n_ctx - cache_n,
                                                &new_tokens,
                                                task.stream && task.return_progress ? send_prefill_progress : nullptr,
                                                &progress);
    if (rc == -2) {
        // Nothing was written, so a cache this request meant to extend is still good.
        if (reuse) {
            e->cached_prompt = std::move(cached_prompt);
            e->cached_images = std::move(cached_images);
        }
        throw context_overflow(static_cast<int64_t>(cache_n) + new_tokens, n_ctx);
    }
    if (rc != 0) {
        throw std::runtime_error(mb_mtmd_get_last_error(e->ctx));
    }
    e->kv_tokens = cache_n + new_tokens;
    const int32_t prompt_tokens = e->kv_tokens;

    stream.delta({{"role", "assistant"}, {"content", nullptr}});

    // Generation, sampled the way llama-server samples.
    llama_context * lctx = mb_mtmd_layla_llama_context(e->ctx);  // a prefill may rebuild it
    std::unique_ptr<common_sampler, void (*)(common_sampler *)> sampler(
        common_sampler_init(model, task.sampling),
        common_sampler_free);
    if (!sampler) {
        throw std::runtime_error("failed to initialise the sampler");
    }
    task_result_state state(task.chat_parser_params);
    const std::vector<std::string> & stops = task.antiprompt;
    size_t max_stop = 0;
    for (const auto & stop : stops) {
        max_stop = std::max(max_stop, stop.size());
    }

    int64_t limit = static_cast<int64_t>(n_ctx) - e->kv_tokens;
    if (task.n_predict >= 0) {
        limit = std::min<int64_t>(limit, task.n_predict);
    }
    std::string text;     // what the parser sees: special tokens only where preserved
    std::string kv_text;  // what the KV cache holds, every token as its text
    size_t sent = 0;      // bytes of `text` handed to the parser so far
    int64_t generated = 0;
    bool model_ended = false;
    bool hit_limit = false;
    for (;;) {
        if (is_cancelled != nullptr && is_cancelled(user_data)) {
            break;
        }
        if (generated >= limit) {
            hit_limit = true;
            break;
        }
        const llama_token token = common_sampler_sample(sampler.get(), lctx, -1);
        common_sampler_accept(sampler.get(), token, true);
        if (llama_vocab_is_eog(vocab, token) && !task.sampling.ignore_eos) {
            // Not decoded, as in llama-server: the next prompt renders the end of turn.
            model_ended = true;
            break;
        }
        const bool special = task.sampling.preserved_tokens.count(token) > 0;
        text += common_token_to_piece(lctx, token, special);
        kv_text += common_token_to_piece(lctx, token, true);
        if (mb_mtmd_layla_decode_token(e->ctx, token) != 0) {
            throw std::runtime_error(mb_mtmd_get_last_error(e->ctx));
        }
        ++generated;
        ++e->kv_tokens;

        if (!stops.empty()) {
            const size_t at = find_stop(text, stops, sent > max_stop ? sent - max_stop : 0);
            if (at != std::string::npos) {
                text.resize(at);
                break;
            }
        }
        // Whole UTF-8 sequences only, and nothing that could still become a stop string.
        const size_t complete = validate_utf8(text);
        const size_t safe = complete - std::min(complete, partial_stop_length(text.substr(0, complete), stops));
        if (safe > sent) {
            std::vector<common_chat_msg_diff> diffs;
            state.update_chat_msg(text.substr(sent, safe - sent), /*is_partial=*/true, diffs);
            stream.diffs(diffs);
            sent = safe;
        }
    }

    std::vector<common_chat_msg_diff> diffs;
    common_chat_msg message = state.update_chat_msg(text.substr(sent), /*is_partial=*/false, diffs);
    stream.diffs(diffs);
    const std::string finish_reason = hit_limit ? "length" : !message.tool_calls.empty() ? "tool_calls" : "stop";

    // Closing the bridge's turn also parks the expert window's memory until the next
    // decode, as the MiniCPM-V app does between replies.
    const mb_mtmd_perf perf = mb_mtmd_perf_snapshot(e->ctx);

    // Only a reply the model ended itself leaves a cache the next request can extend.
    if (model_ended) {
        e->cached_prompt = prompt + kv_text;
        e->cached_images = std::move(image_keys);
    }

    const json usage = {{"prompt_tokens", prompt_tokens},
                        {"completion_tokens", generated},
                        {"total_tokens", e->kv_tokens}};
    const json timings = timings_json(perf, cache_n);
    stream.send({{"choices", json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}})},
                 {"timings", timings}});
    if (task.include_usage) {
        stream.send({{"choices", json::array()}, {"usage", usage}, {"timings", timings}});
    }

    json reply = message.to_json_oaicompat();
    reply["role"] = "assistant";
    return dump({
        {"id", stream.id},
        {"object", "chat.completion"},
        {"created", stream.created},
        {"model", kModelName},
        {"choices", json::array({{{"index", 0}, {"message", reply}, {"finish_reason", finish_reason}}})},
        {"usage", usage},
        {"timings", timings},
    });
}

// The model's chat template read from its GGUF without loading any weights.
struct template_source {
    std::unique_ptr<llama_model, void (*)(llama_model *)> model{nullptr, llama_model_free};
    server_chat_params chat;
};

void load_template(const char * model_path, template_source & out) {
    if (model_path == nullptr || model_path[0] == '\0') {
        throw std::runtime_error("no model file");
    }
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    out.model.reset(llama_model_load_from_file(model_path, params));
    if (!out.model) {
        throw std::runtime_error(std::string("cannot read the model file ") + model_path);
    }
    configure_chat(out.chat, common_chat_templates_init(out.model.get(), ""));
    if (!out.chat.tmpls) {
        throw std::runtime_error("the model has no chat template");
    }
}

// A saved conversation's record, packed into the int32 token list the bridge stores with
// the KV cache: magic, version, kv_tokens, the number of strings (the cached prompt, then
// one key per image), each one's byte length, and then their bytes, padded to whole
// int32s. The magic keeps any other engine's sequence state from passing for one.
constexpr int32_t kStateMagic = 0x4C4D4356;  // "LMCV"
constexpr int32_t kStateVersion = 1;

std::vector<int32_t> pack_record(const minicpm_engine & e) {
    std::vector<const std::string *> strings{&e.cached_prompt};
    for (const auto & key : e.cached_images) {
        strings.push_back(&key);
    }
    std::vector<int32_t> out{kStateMagic, kStateVersion, e.kv_tokens, static_cast<int32_t>(strings.size())};
    std::string bytes;
    for (const std::string * s : strings) {
        if (s->size() > static_cast<size_t>(INT32_MAX)) {
            throw std::runtime_error("the conversation is too large to save");
        }
        out.push_back(static_cast<int32_t>(s->size()));
        bytes += *s;
    }
    bytes.resize((bytes.size() + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t), '\0');
    const size_t head = out.size();
    out.resize(head + bytes.size() / sizeof(int32_t));
    std::memcpy(out.data() + head, bytes.data(), bytes.size());
    return out;
}

// False when meta is not a record this version wrote.
bool unpack_record(const int32_t * meta, size_t n, minicpm_engine & e) {
    if (n < 5 || meta[0] != kStateMagic || meta[1] != kStateVersion || meta[2] <= 0 || meta[3] < 1) {
        return false;
    }
    const size_t n_strings = static_cast<size_t>(meta[3]);
    if (n_strings > n - 4) {
        return false;
    }
    const char * bytes = reinterpret_cast<const char *>(meta + 4 + n_strings);
    const size_t available = (n - 4 - n_strings) * sizeof(int32_t);
    std::vector<std::string> strings;
    size_t at = 0;
    for (size_t i = 0; i < n_strings; ++i) {
        if (meta[4 + i] < 0 || static_cast<size_t>(meta[4 + i]) > available - at) {
            return false;
        }
        strings.emplace_back(bytes + at, static_cast<size_t>(meta[4 + i]));
        at += static_cast<size_t>(meta[4 + i]);
    }
    if (strings[0].empty()) {
        return false;
    }
    e.kv_tokens = meta[2];
    e.cached_prompt = std::move(strings[0]);
    e.cached_images.assign(std::make_move_iterator(strings.begin() + 1), std::make_move_iterator(strings.end()));
    return true;
}

void write_error(char * error, size_t error_size, const char * message) {
    if (error != nullptr && error_size > 0) {
        std::snprintf(error, error_size, "%s", message);
    }
}

}  // namespace

// Called by minicpm_load once the bridge is up.
bool minicpm_init_chat(minicpm_engine * engine, std::string & error) {
    common_chat_templates_ptr tmpls = common_chat_templates_init(mb_mtmd_layla_model(engine->ctx), "");
    if (!tmpls) {
        error = "the model has no chat template";
        return false;
    }
    configure_chat(engine->chat, std::move(tmpls));
    engine->kv_tokens = 0;
    return true;
}

extern "C" {

char * minicpm_chat_completions(minicpm_engine * engine,
                                const char * request_json,
                                minicpm_chunk_callback on_chunk,
                                minicpm_cancel_callback is_cancelled,
                                void * user_data) {
    if (engine == nullptr) {
        return nullptr;
    }
    engine->last_error.clear();
    try {
        return copy_string(serve(engine, request_json, on_chunk, is_cancelled, user_data));
    } catch (const std::exception & error) {
        engine->last_error = error.what();
    } catch (...) {
        engine->last_error = "unknown MiniCPM error";
    }
    if (engine->last_error.empty()) {
        engine->last_error = "MiniCPM chat completion failed";
    }
    // Close the bridge's turn as a finished one would, which releases the expert window.
    mb_mtmd_perf_snapshot(engine->ctx);
    return nullptr;
}

char * minicpm_render_prompt(const char * model_path, const char * request_json, char * error, size_t error_size) {
    try {
        template_source source;
        load_template(model_path, source);
        json body = json::parse(request_json != nullptr ? request_json : "");
        apply_reasoning_effort(body);
        std::vector<raw_buffer> images;
        const json data = oaicompat_chat_params_parse(body, source.chat, images);
        return copy_string(with_mtmd_marker(data.at("prompt").get<std::string>()));
    } catch (const std::exception & e) {
        write_error(error, error_size, e.what());
    }
    return nullptr;
}

bool minicpm_read_template_caps(const char * model_path,
                                minicpm_template_caps * caps,
                                char * error,
                                size_t error_size) {
    try {
        template_source source;
        load_template(model_path, source);
        const std::map<std::string, bool> found = common_chat_templates_get_caps(source.chat.tmpls.get());
        auto cap = [&](const char * name) {
            const auto it = found.find(name);
            return it != found.end() && it->second;
        };
        caps->supports_tools = cap("supports_tools");
        caps->supports_tool_calls = cap("supports_tool_calls");
        caps->supports_parallel_tool_calls = cap("supports_parallel_tool_calls");
        caps->supports_system_role = cap("supports_system_role");
        caps->supports_preserve_reasoning = cap("supports_preserve_reasoning");
        caps->supports_reasoning_effort = cap("supports_reasoning_effort");
        caps->supports_string_content = cap("supports_string_content");
        caps->supports_typed_content = cap("supports_typed_content");
        caps->supports_object_arguments = cap("supports_object_arguments");
        return true;
    } catch (const std::exception & e) {
        write_error(error, error_size, e.what());
    }
    return false;
}

int64_t minicpm_save_state(minicpm_engine * engine, const char * path) {
    if (engine == nullptr) {
        return -1;
    }
    engine->last_error.clear();
    try {
        if (path == nullptr || path[0] == '\0') {
            throw std::runtime_error("no state file path");
        }
        if (engine->cached_prompt.empty() || engine->kv_tokens <= 0) {
            throw std::runtime_error("the KV cache holds no finished reply to resume from");
        }
        const std::vector<int32_t> record = pack_record(*engine);
        const size_t written = mb_mtmd_layla_state_save(engine->ctx, path, record.data(), record.size());
        if (written == 0) {
            throw std::runtime_error(mb_mtmd_get_last_error(engine->ctx));
        }
        return static_cast<int64_t>(written);
    } catch (const std::exception & error) {
        engine->last_error = error.what();
    }
    return -1;
}

int32_t minicpm_restore_state(minicpm_engine * engine, const char * path) {
    if (engine == nullptr) {
        return -1;
    }
    engine->last_error.clear();
    // Whatever the cache held is gone from here on, restored or not.
    engine->cached_prompt.clear();
    engine->cached_images.clear();
    engine->kv_tokens = 0;
    try {
        if (path == nullptr || path[0] == '\0') {
            throw std::runtime_error("no state file path");
        }
        int32_t * meta = nullptr;
        size_t n_meta = 0;
        if (mb_mtmd_layla_state_load(engine->ctx, path, &meta, &n_meta) == 0) {
            throw std::runtime_error(mb_mtmd_get_last_error(engine->ctx));
        }
        std::unique_ptr<int32_t, void (*)(void *)> owned(meta, std::free);
        if (!unpack_record(meta, n_meta, *engine) || engine->kv_tokens > mb_mtmd_layla_n_ctx(engine->ctx)) {
            engine->cached_prompt.clear();
            engine->cached_images.clear();
            engine->kv_tokens = 0;
            mb_mtmd_clean_kv_cache(engine->ctx);
            throw std::runtime_error(std::string("not a MiniCPM conversation state: ") + path);
        }
        return engine->kv_tokens;
    } catch (const std::exception & error) {
        engine->last_error = error.what();
    }
    return -1;
}

void minicpm_string_free(char * s) {
    std::free(s);
}

}  // extern "C"
