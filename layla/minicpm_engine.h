#pragma once

// The engine behind minicpm_capi.h, shared by minicpm_capi.cpp (loading) and
// minicpm_oai.cpp (serving requests). Not part of the C ABI.

#include "minicpm_capi.h"

#include "MBMtmd.h"
#include "server-common.h"

#include <cstdint>
#include <string>
#include <vector>

struct minicpm_engine {
    mb_mtmd_context * ctx;
    minicpm_load_info info;
    std::string last_error;

    // How llama-server would turn a request into a prompt for this model: its own
    // chat template from the GGUF, Jinja, images allowed.
    server_chat_params chat;

    // What the KV cache holds, when a request may extend it: the rendered prompt it was
    // prefilled from plus every generated token, as text, and the images in it. Empty
    // whenever the cache holds anything else (an unfinished reply, a failed request),
    // which makes the next request start over.
    std::string cached_prompt;
    std::vector<std::string> cached_images;
    // Tokens in the KV cache. The bridge counts positions, which images take fewer of.
    int32_t kv_tokens;
};

// Sets up the chat template from the loaded model (minicpm_oai.cpp). False, with the
// reason in error, when the model has none.
bool minicpm_init_chat(minicpm_engine * engine, std::string & error);
