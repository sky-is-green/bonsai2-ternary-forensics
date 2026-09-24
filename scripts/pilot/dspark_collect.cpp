// DSpark/DFlash2 feature collector: dump intermediate target-layer input hidden
// states for draft-model distillation.
//
// Adapted from ProCreations' `training/collect.cpp` (which dumps the MTP/final
// hidden via `llama_get_embeddings_nextn`).  DFlash2/DSpark consume *intermediate*
// target layers (`target_layer_ids`), which the fork exposes through
// `llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`.
//
// Requires the PrismML-Eng/llama.cpp fork (the layer-input API is fork-only).
// Build (after building the fork):
//
//   g++ -O2 -std=c++17 scripts/pilot/dspark_collect.cpp \
//       -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
//       -I"$LLAMA/common" -I"$LLAMA/vendor" \
//       -L"$LLAMA/build/bin" -lllama -lggml -lggml-base \
//       -Wl,-rpath,"$LLAMA/build/bin" -o dspark_collect
//
// Run:
//
//   ./dspark_collect base.gguf texts.jsonl outdir 5,19,33,47,61
//
// Each input line is `{"id": "...", "text": "..."}`.  For every sequence it
// writes `<id>.tokens.i32`, `<id>.L<layer>.hidden.f16` and `<id>.json`, plus an
// append-only `index.jsonl`.  Hidden rows are `[n_tokens, n_embd]` fp16; verify
// the row/column order on one prompt against a known-good extraction before a
// large run (the fork stores layer inputs as a flat `n_embd * n_batch` view).

#include "llama.h"
#include "llama-ext.h"  // fork staging API: llama_{set,get}_embeddings_layer_inp
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "usage: dspark_collect <model.gguf> <texts.jsonl> <outdir> <layer_ids>\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string outdir = argv[3];

    std::vector<uint32_t> layers;
    {
        std::stringstream ss(argv[4]);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) layers.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
    }
    if (layers.empty()) { std::cerr << "no layer ids\n"; return 2; }

    std::filesystem::create_directories(outdir);

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) { std::cerr << "model load failed\n"; return 3; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 1024;
    cparams.n_ubatch = 512;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 16;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { std::cerr << "context init failed\n"; return 4; }

    for (uint32_t layer : layers) {
        llama_set_embeddings_layer_inp(ctx, layer, true);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    for (uint32_t layer : layers) {
        if (layer > static_cast<uint32_t>(n_layer)) {
            std::cerr << "layer " << layer << " exceeds n_layer " << n_layer << "\n";
            return 2;
        }
    }

    llama_batch batch = llama_batch_init(1024, 0, 1);
    std::ifstream fin(input_path);
    std::ofstream index(outdir + "/index.jsonl", std::ios::app);
    std::string line;
    int count = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        json item = json::parse(line);
        const std::string text = item.at("text").get<std::string>();
        const std::string id = item.at("id").get<std::string>();

        std::vector<llama_token> tokens(text.size() + 32);
        int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()),
                               tokens.data(), static_cast<int>(tokens.size()), true, true);
        if (n < 0) { std::cerr << "tokenize failed for " << id << "\n"; return 5; }
        n = std::min(n, 1024);
        if (n < 64) continue;
        tokens.resize(n);

        if (std::filesystem::exists(outdir + "/" + id + ".done")) continue;

        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n - 1);
        }
        if (llama_decode(ctx, batch) != 0) { std::cerr << "decode failed for " << id << "\n"; return 6; }

        json meta;
        meta["id"] = id;
        meta["tokens"] = n;
        meta["hidden_dim"] = n_embd;
        meta["layers"] = layers;

        for (uint32_t layer : layers) {
            const float * hidden = llama_get_embeddings_layer_inp(ctx, layer);
            if (hidden == nullptr) { std::cerr << "no layer input for " << layer << "\n"; return 7; }
            std::vector<ggml_fp16_t> half(static_cast<size_t>(n) * n_embd);
            ggml_fp32_to_fp16_row(hidden, half.data(), static_cast<int64_t>(half.size()));
            std::ofstream out(outdir + "/" + id + ".L" + std::to_string(layer) + ".hidden.f16",
                              std::ios::binary);
            out.write(reinterpret_cast<const char *>(half.data()),
                      static_cast<std::streamsize>(half.size() * sizeof(ggml_fp16_t)));
        }

        std::ofstream tokens_out(outdir + "/" + id + ".tokens.i32", std::ios::binary);
        tokens_out.write(reinterpret_cast<const char *>(tokens.data()),
                         static_cast<std::streamsize>(n * sizeof(llama_token)));
        std::ofstream(outdir + "/" + id + ".json") << meta.dump();
        std::ofstream(outdir + "/" + id + ".done") << "";
        index << meta.dump() << "\n";
        index.flush();
        count++;
        if (count % 8 == 0) std::cout << json({{"samples", count}}).dump() << std::endl;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "done " << count << std::endl;
    return 0;
}
