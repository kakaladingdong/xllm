/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "autoencoder_kl_flux2.h"
// #include "core/framework/chat_template/jinja_chat_template.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/dit/flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"
#include "models/llm/npu/mistral3.h"
#include "transformer_flux2.h"

namespace xllm {

inline std::string SYSTEM_MESSAGE =
    "You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object attribution and "
    "actions "
    "without speculation.";

inline std::string SYSTEM_MESSAGE_UPSAMPLING_T2I =
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their "
    "core subject and intent.\n"
    "Guidelines:\n"
    "1. Structure: Keep structured inputs structured (enhance within fields). "
    "Convert natural language to detailed paragraphs.\n"
    "2. Details: Add concrete visual specifics - form, scale, textures, "
    "materials, lighting (quality, direction, color), shadows, spatial "
    "relationships, and environmental context.\n"
    "3. Text in Images: Put ALL text in quotation marks, matching the "
    "prompt's language. Always provide explicit quoted text for objects that "
    "would "
    "contain text in reality (signs, labels, screens, etc.) - without it, "
    "the model generates gibberish.\n"
    "Output only the revised prompt and nothing else.";

inline std::string SYSTEM_MESSAGE_UPSAMPLING_I2I =
    "You are FLUX.2 by Black Forest Labs, an image-editing expert. You "
    "convert editing requests into one concise instruction (50-80 words, ~30 "
    "for "
    "brief requests).\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid \"whimsical,\" \"cascading,\" "
    "etc.)\n"
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Reference actual image elements\n"
    "- Turn negatives into positives (\"don't change X\" → \"keep X\")\n"
    "- Make abstractions concrete (\"futuristic\" → \"glowing cyan neon, "
    "metallic "
    "panels\")\n"
    "- Keep content PG-13\n"
    "Output only the final instruction in plain text and nothing else.";

inline std::vector<std::vector<std::unordered_map<std::string, std::string>>>
format_input(const std::vector<std::string>& prompts,
             const std::string& system_message = SYSTEM_MESSAGE,
             const std::vector<std::vector<torch::Tensor>>& images =
                 std::vector<std::vector<torch::Tensor>>()) {
  std::vector<std::vector<std::unordered_map<std::string, std::string>>>
      messages_batch;
  messages_batch.reserve(prompts.size());

  for (const auto& prompt : prompts) {
    std::vector<std::unordered_map<std::string, std::string>> messages;

    messages.push_back(
        {{"role", "system"},
         {"content",
          "[{\"type\": \"text\", \"text\": \"" + system_message + "\"}]"}});

    messages.push_back(
        {{"role", "user"},
         {"content", "[{\"type\": \"text\", \"text\": \"" + prompt + "\"}]"}});

    messages_batch.push_back(messages);
  }

  return messages_batch;
}

float compute_empirical_mu(int64_t image_seq_len, int64_t num_steps) {
  double a1 = 8.73809524e-05, b1 = 1.89833333;
  double a2 = 0.00016927, b2 = 0.45666666;

  double mu;
  if (image_seq_len > 4300) {
    mu = a2 * image_seq_len + b2;
    return static_cast<float>(mu);
  }

  double m_200 = a2 * image_seq_len + b2;
  double m_10 = a1 * image_seq_len + b1;

  double a = (m_200 - m_10) / 190.0;
  double b = m_200 - 200.0 * a;
  mu = a * num_steps + b;

  return static_cast<float>(mu);
}

std::pair<torch::Tensor, int64_t> flux2_retrieve_timesteps(
    FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

class Flux2PosEmbedImpl : public torch::nn::Module {
 public:
  Flux2PosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    auto seq_len = txt_ids.size(0);

    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      LOG(INFO) << "before cat txt_ids shape, img_ids shape" << txt_ids.sizes()
                << img_ids.sizes();
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 1);
      LOG(INFO) << "----------concat.ids" << ids.sizes();
      cached_image_height_ = height;
      cached_image_width_ = width;
      max_seq_len_ = seq_len;
      auto [cos, sin] = forward(ids);
      LOG(INFO) << "----------C++.forward.ids" << ids.sizes();
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
  }
  /*
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
      int64_t n_axes = axes_dim_.size();
      std::vector<torch::Tensor> cos_out, sin_out;
      auto pos = ids.to(torch::kFloat32);
      torch::Dtype freqs_dtype = torch::kFloat64;
      for (int64_t i = 0; i < n_axes; ++i) {
        auto pos_slice = pos.select(-1, i);
        auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                              pos_slice,
                                              theta_,
                                              true,  // repeat_interleave_real
                                              1,
                                              1,
                                              true,  // use_real
                                              freqs_dtype);
        auto cos = result[0];
        auto sin = result[1];
        cos_out.push_back(cos);
        sin_out.push_back(sin);
      }

      auto freqs_cos = torch::cat(cos_out, -1);
      auto freqs_sin = torch::cat(sin_out, -1);
      return {freqs_cos, freqs_sin};
    }
  */
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = axes_dim_.size();
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    torch::Dtype freqs_dtype = torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      LOG(INFO) << "----------before---pos.select()" << std::endl;
      auto pos_slice = pos.select(-1, i).squeeze(0);  // 已修改
      LOG(INFO) << "----------after---pos.select(-1, i)" << std::endl;
      auto result = get_1d_rotary_pos_embed(
          axes_dim_[i], pos_slice, theta_, true, 1, 1, true, freqs_dtype);
      LOG(INFO) << "----------get_1d_rotary_pos_embed-----" << std::endl;
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
      LOG(INFO) << "----------for_loop_end" << std::endl;
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);
    LOG(INFO) << "----------end_forward" << std::endl;
    return {freqs_cos, freqs_sin};
  }

 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t max_seq_len_ = -1;
  int64_t cached_image_height_ = -1;
  int64_t cached_image_width_ = -1;
};
TORCH_MODULE(Flux2PosEmbed);

class Flux2PipelineBaseImpl : public torch::nn::Module {
 protected:
  torch::Tensor get_mistral3_prompt_embeds(
      std::vector<std::string>& prompt,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512,
      const std::vector<int64_t>& hidden_states_layers = {10, 20, 30},
      const std::string& system_message = SYSTEM_MESSAGE) {
// ===================== 1. 前置检查 =====================
    CHECK(tokenizer_ != nullptr) << "Tokenizer not initialized!";
    CHECK(!mistral3_.is_empty()) << "Mistral3 model not loaded!";
    if (prompt.empty()) {
        LOG(WARNING) << "Empty prompt list, using empty string as default";
        prompt = {""};
    }
    int64_t batch_size = prompt.size();

    // ===================== 2. 格式化Prompt =====================
    auto messages_batch = format_input(prompt, system_message);
    std::vector<std::string> formatted_prompts;
    formatted_prompts.reserve(batch_size);

    for (const auto& messages : messages_batch) {
        auto formatted = apply_chat_template(messages);
        if (!formatted.has_value()) {
            throw std::runtime_error("Failed to apply Mistral3 chat template");
        }
        formatted_prompts.push_back(formatted.value());
    }

    // ===================== 3. 生成Mistral3格式Input ID =====================
    // 3.1 批量编码
    std::vector<std::vector<int32_t>> batch_token_ids;
    batch_token_ids.reserve(prompt.size());
    CHECK(tokenizer_->batch_encode(formatted_prompts, &batch_token_ids)) 
        << "Mistral3 tokenizer batch encode failed!";

    // 3.2 确定PAD Token ID
    int32_t pad_token_id = MISTRAL3_DEFAULT_PAD_ID;
    const auto& text_encoder_args = context_.get_model_args("text_encoder");
    if (text_encoder_args.pad_token_id() > 0) {
        pad_token_id = text_encoder_args.pad_token_id();
    } else if (auto eot_id = tokenizer_->token_to_id("<|endoftext|>"); eot_id.has_value()) {
        pad_token_id = eot_id.value();
    }

    // 3.3 截断/填充（左对齐，末尾PAD）
    std::vector<int32_t> input_ids_flat;
    std::vector<int64_t> orig_seq_lens;
    input_ids_flat.reserve(batch_size * max_sequence_length);
    orig_seq_lens.reserve(batch_size);

    for (auto& token_ids : batch_token_ids) {
        int64_t orig_len = token_ids.size();
        orig_seq_lens.push_back(orig_len);

        if (orig_len > max_sequence_length) {
            LOG(WARNING) << "Prompt truncated from " << orig_len << " to " << max_sequence_length;
            token_ids.resize(max_sequence_length);
            orig_len = max_sequence_length;
        }
        int64_t pad_len = max_sequence_length - orig_len;
        if (pad_len > 0) {
            token_ids.insert(token_ids.end(), pad_len, pad_token_id);
        }
        input_ids_flat.insert(input_ids_flat.end(), token_ids.begin(), token_ids.end());
    }

    // 3.4 构建1D Input ID张量（Mistral3 forward要求）
    torch::Tensor tokens = torch::tensor(input_ids_flat, torch::kLong).to(options_.device());

    // ===================== 4. 生成Mistral3格式Attention Mask =====================
    // 4.1 生成1D扁平mask
    std::vector<int32_t> attn_mask_flat;
    std::vector<int> q_seq_lens_vec, kv_seq_lens_vec;
    attn_mask_flat.reserve(batch_size * max_sequence_length);
    q_seq_lens_vec.reserve(batch_size);
    kv_seq_lens_vec.reserve(batch_size);

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t orig_len = orig_seq_lens[b];
        for (int64_t j = 0; j < max_sequence_length; ++j) {
            attn_mask_flat.push_back(j < orig_len ? 1 : 0);
        }
        q_seq_lens_vec.push_back(static_cast<int>(max_sequence_length));
        kv_seq_lens_vec.push_back(static_cast<int>(max_sequence_length));
    }

    torch::Tensor attention_mask = torch::tensor(attn_mask_flat, torch::kLong)
        .view({batch_size, max_sequence_length})
        .to(options_.device());

    // ===================== 5. 生成Position张量（适配Mistral3 RoPE） =====================
    auto mask_flat = attention_mask.view({-1});
    auto positions_1d = mask_flat.to(torch::kInt64).cumsum(-1) - 1;
    positions_1d = positions_1d.masked_fill(mask_flat == 0, 1);
    torch::Tensor positions = positions_1d.to(options_.device());

    // ===================== 6. 构建ModelInputParams（适配Chunked Prefill） =====================
    ModelInputParams input_params;
    input_params.q_max_seq_len = max_sequence_length;
    input_params.kv_max_seq_len = max_sequence_length;
    input_params.batch_forward_type = BatchForwardType::PREFILL;
    input_params.num_sequences = static_cast<int>(batch_size);
    input_params.q_seq_lens_vec = q_seq_lens_vec;
    input_params.kv_seq_lens_vec = kv_seq_lens_vec;

    // 构建累积序列长度张量
    std::vector<int> cu_seq_lens = {0};
    int cum_len = 0;
    for (int len : q_seq_lens_vec) {
        cum_len += len;
        cu_seq_lens.push_back(cum_len);
    }
    input_params.q_seq_lens = torch::tensor(cu_seq_lens, torch::kInt).to(tokens.device());
    input_params.kv_seq_lens = input_params.q_seq_lens;

    // 传递Attention Mask
    input_params.graph_buffer.attn_mask = attention_mask.view({-1}).to(torch::kFloat32);
    input_params.input_embedding = torch::Tensor();

    // ===================== 7. 生成KV Cache =====================
    std::vector<KVCache> kv_caches;
    int64_t num_layers = text_encoder_args.n_layers();
    kv_caches.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
        kv_caches.emplace_back(torch::Tensor(), torch::Tensor());
    }

    // ===================== 8. 调用Mistral3 Forward =====================
    ModelOutput model_output = mistral3_->forward(tokens, positions, kv_caches, input_params);
    torch::Tensor hidden_states_flat = model_output.hidden_states;

    // ===================== 9. 处理输出Embedding =====================
    CHECK(hidden_states_flat.dim() == 2) << "Mistral3 forward output must be 2D [total_seq_len, hidden_size]";
    int64_t hidden_size = hidden_states_flat.size(-1);

    // 重塑为[batch_size, max_sequence_length, hidden_size]
    torch::Tensor hidden_states = hidden_states_flat.view({
        batch_size, max_sequence_length, hidden_size
    });

    // 适配多图生成
    auto prompt_embeds = hidden_states.reshape({batch_size, max_sequence_length, hidden_size});
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds = prompt_embeds.view({
        batch_size * num_images_per_prompt, max_sequence_length, hidden_size
    });

    // ===================== 10. 返回最终Embedding =====================
    return prompt_embeds.to(options_);
  }

  torch::Tensor prepare_text_ids(const torch::Tensor& prompt_embeds) {
    int64_t batch_size = prompt_embeds.size(0);
    int64_t seq_len = prompt_embeds.size(1);

    std::vector<torch::Tensor> out_ids;
    out_ids.reserve(batch_size);

    for (int64_t i = 0; i < batch_size; ++i) {
      auto t = torch::arange(1, options_);
      auto h = torch::arange(1, options_);
      auto w = torch::arange(1, options_);
      auto l = torch::arange(seq_len, options_);

      auto grid = torch::meshgrid({t, h, w, l}, "ij");
      auto coords = torch::stack({grid[0].flatten(),
                                  grid[1].flatten(),
                                  grid[2].flatten(),
                                  grid[3].flatten()},
                                 -1);
      out_ids.push_back(coords);
    }

    auto text_ids = torch::stack(out_ids, 0);
    return text_ids;
  }

  std::tuple<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<torch::Tensor> prompt_embeds,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512,
      const std::vector<int64_t>& hidden_states_layers = {10, 20, 30},
      const std::string& system_message = SYSTEM_MESSAGE) {
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }
    if (!prompt_embeds.has_value()) {
      prompt_embeds = get_mistral3_prompt_embeds(prompt_list,
                                                 num_images_per_prompt,
                                                 max_sequence_length,
                                                 hidden_states_layers);
    }
    torch::Tensor text_ids = prepare_text_ids(prompt_embeds.value());

    return std::make_tuple(prompt_embeds.value(), text_ids);
  }

  torch::Tensor prepare_latent_image_ids(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    auto t = torch::arange(1, 2, options_);
    auto h = torch::arange(height, options_);
    auto w = torch::arange(width, options_);
    auto l = torch::arange(1, 2, options_);

    auto grid = torch::meshgrid({t, h, w, l}, "ij");
    auto coords = torch::stack({grid[0].flatten(),
                                grid[1].flatten(),
                                grid[2].flatten(),
                                grid[3].flatten()},
                               -1);

    auto latent_image_ids = coords.unsqueeze(0).expand({batch_size, -1, -1});
    return latent_image_ids;
  }

  torch::Tensor pack_latents(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    torch::Tensor latents_packed =
        latents.reshape({batch_size, num_channels, height * width});
    latents_packed = latents_packed.permute({0, 2, 1});

    return latents_packed;
  }

  torch::Tensor patchify_latents(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels_latents = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    torch::Tensor latents_patched = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    latents_patched = latents_patched.permute({0, 1, 3, 5, 2, 4});
    latents_patched = latents_patched.reshape(
        {batch_size, num_channels_latents * 4, height / 2, width / 2});

    return latents_patched;
  }

  torch::Tensor unpatchify_latents(const torch::Tensor& latents) {
    int64_t batch_size = latents.size(0);
    int64_t num_channels_latents = latents.size(1);
    int64_t height = latents.size(2);
    int64_t width = latents.size(3);

    torch::Tensor latents_unpatched = latents.reshape(
        {batch_size, num_channels_latents / (2 * 2), 2, 2, height, width});
    latents_unpatched = latents_unpatched.permute({0, 1, 4, 2, 5, 3});
    latents_unpatched = latents_unpatched.reshape(
        {batch_size, num_channels_latents / (2 * 2), height * 2, width * 2});

    return latents_unpatched;
  }

  torch::Tensor unpack_latents(const torch::Tensor& latents,
                               int64_t height,
                               int64_t width,
                               int64_t vae_scale_factor) {
    int64_t batch_size = latents.size(0);
    int64_t num_patches = latents.size(1);
    int64_t channels = latents.size(2);
    height = 2 * (height / (vae_scale_factor_ * 2));
    width = 2 * (width / (vae_scale_factor_ * 2));

    torch::Tensor latents_unpacked =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents_unpacked = latents_unpacked.permute({0, 3, 1, 4, 2, 5});
    latents_unpacked = latents_unpacked.reshape(
        {batch_size, channels / (2 * 2), height, width});

    return latents_unpacked;
  }

  torch::Tensor unpack_latents_with_ids(const torch::Tensor& latents,
                                        const torch::Tensor& latent_ids) {
    int64_t batch_size = latents.size(0);
    int64_t seq_len = latents.size(1);
    int64_t channels = latents.size(2);

    std::vector<torch::Tensor> x_list;
    for (int64_t i = 0; i < batch_size; ++i) {
      torch::Tensor data = latents[i];
      torch::Tensor pos = latent_ids[i];

      torch::Tensor h_ids = pos.select(1, 1).to(torch::kInt64);
      torch::Tensor w_ids = pos.select(1, 2).to(torch::kInt64);

      int64_t h = h_ids.max().item<int64_t>() + 1;
      int64_t w = w_ids.max().item<int64_t>() + 1;

      torch::Tensor flat_ids = h_ids * w + w_ids;

      torch::Tensor out = torch::zeros({h * w, channels}, data.options());
      out.scatter_(0, flat_ids.unsqueeze(1).expand({-1, channels}), data);

      out = out.view({h, w, channels}).permute({2, 0, 1});
      x_list.push_back(out);
    }

    return torch::stack(x_list, 0);
  }

  torch::Tensor _prepare_image_ids(
      const std::vector<torch::Tensor>& image_latents,
      int64_t scale = 10) {
    if (image_latents.empty()) {
      throw std::invalid_argument("image_latents list cannot be empty!");
    }

    int64_t num_latents = image_latents.size();
    torch::Tensor t_indices = torch::arange(0, num_latents, torch::kInt64);
    torch::Tensor t_coords = scale + scale * t_indices;
    t_coords = t_coords.unsqueeze(1);

    std::vector<torch::Tensor> image_latent_ids;
    for (int64_t i = 0; i < num_latents; ++i) {
      torch::Tensor x = image_latents[i];
      x = x.squeeze(0);
      if (x.dim() != 3) {
        throw std::invalid_argument(
            "Each image latent must be 3D (C, H, W) or 4D (1, C, H, W), got " +
            std::to_string(x.dim()) + "D tensor!");
      }

      int64_t height = x.size(1);
      int64_t width = x.size(2);

      torch::Tensor h_coords = torch::arange(0, height, torch::kInt64);
      torch::Tensor w_coords = torch::arange(0, width, torch::kInt64);
      torch::Tensor l_coords = torch::arange(0, 1, torch::kInt64);

      torch::Tensor t = t_coords[i].expand({1});
      auto t_exp = t.repeat({height * width * 1});
      auto h_exp = h_coords.repeat_interleave(width * 1).repeat({1});
      auto w_exp = w_coords.repeat({height}).repeat_interleave(1);
      auto l_exp = l_coords.repeat({height * width}).repeat({1});

      torch::Tensor coords = torch::stack({t_exp, h_exp, w_exp, l_exp}, 1);
      image_latent_ids.push_back(coords);
    }

    torch::Tensor combined_coords = torch::cat(image_latent_ids, 0);
    combined_coords = combined_coords.unsqueeze(0);
    return combined_coords;
  }

 protected:
  Mistral3ForConditionalGeneration mistral3_{nullptr};
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  std::unique_ptr<Tokenizer> tokenizer_;
  // std::unique_ptr<JinjaChatTemplate> chat_template_;
  torch::TensorOptions options_;
  int tokenizer_max_length_;
  int vae_scale_factor_;
};
}  // namespace xllm
