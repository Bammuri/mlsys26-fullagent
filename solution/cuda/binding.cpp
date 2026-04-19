#include <torch/extension.h>

#include <cstdlib>
#include <tuple>

namespace {
constexpr int64_t kHeadSize = 128;
constexpr int64_t kNumQHeads = 4;
constexpr int64_t kNumKHeads = 4;
constexpr int64_t kNumVHeads = 8;

void check_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& state,
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& b,
    const torch::Tensor& cu_seqlens) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
  TORCH_CHECK(A_log.is_cuda() && a.is_cuda() && dt_bias.is_cuda() && b.is_cuda(), "A_log/a/dt_bias/b must be CUDA tensors");
  TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bfloat16");
  TORCH_CHECK(A_log.scalar_type() == torch::kFloat32, "A_log must be float32");
  TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32, "dt_bias must be float32");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt64, "cu_seqlens must be int64");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be rank-3");
  TORCH_CHECK(q.size(1) == kNumQHeads && k.size(1) == kNumKHeads && v.size(1) == kNumVHeads, "unexpected head counts");
  TORCH_CHECK(q.size(2) == kHeadSize && k.size(2) == kHeadSize && v.size(2) == kHeadSize, "head size must be 128");
  TORCH_CHECK(a.size(0) == q.size(0) && a.size(1) == kNumVHeads, "a must have shape [T, 8]");
  TORCH_CHECK(b.size(0) == q.size(0) && b.size(1) == kNumVHeads, "b must have shape [T, 8]");
  TORCH_CHECK(A_log.numel() == kNumVHeads && dt_bias.numel() == kNumVHeads, "A_log and dt_bias must have 8 elements");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens must be [N+1]");
  if (state.has_value() && state.value().defined()) {
    TORCH_CHECK(state.value().is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.value().scalar_type() == torch::kFloat32, "state must be float32");
  }
}

torch::Tensor softplus_stable(const torch::Tensor& x) {
  return torch::where(x > 20.0, x, torch::log1p(torch::exp(x)));
}

std::tuple<torch::Tensor, torch::Tensor> run_oracle_impl(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale) {
  if (scale == 0.0) {
    scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
  }

  auto device = q.device();
  auto total_seq_len = q.size(0);
  auto num_seqs = cu_seqlens.size(0) - 1;

  auto x = a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32);
  auto g = torch::exp(-torch::exp(A_log.to(torch::kFloat32)) * softplus_stable(x));
  auto beta = torch::sigmoid(b.to(torch::kFloat32));

  auto q_exp = q.repeat_interleave(kNumVHeads / kNumQHeads, 1).to(torch::kFloat32);
  auto k_exp = k.repeat_interleave(kNumVHeads / kNumKHeads, 1).to(torch::kFloat32);
  auto v_f = v.to(torch::kFloat32);

  auto output = torch::empty({total_seq_len, kNumVHeads, kHeadSize}, q.options());
  auto new_state = torch::empty(
      {num_seqs, kNumVHeads, kHeadSize, kHeadSize},
      torch::TensorOptions().device(device).dtype(torch::kFloat32));

  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    int64_t seq_start = cu_seqlens[seq_idx].item<int64_t>();
    int64_t seq_end = cu_seqlens[seq_idx + 1].item<int64_t>();

    torch::Tensor state_hkv;
    if (state.has_value() && state.value().defined()) {
      state_hkv = state.value()[seq_idx].to(torch::kFloat32).transpose(-1, -2).contiguous();
    } else {
      state_hkv = torch::zeros({kNumVHeads, kHeadSize, kHeadSize}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    }

    for (int64_t t = seq_start; t < seq_end; ++t) {
      auto q_h1k = q_exp[t].unsqueeze(1);
      auto k_h1k = k_exp[t].unsqueeze(1);
      auto v_h1v = v_f[t].unsqueeze(1);
      auto g_h11 = g[t].unsqueeze(1).unsqueeze(2);
      auto beta_h11 = beta[t].unsqueeze(1).unsqueeze(2);

      auto old_state = g_h11 * state_hkv;
      auto old_v = torch::matmul(k_h1k, old_state);
      auto new_v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v;
      auto state_remove = torch::matmul(k_h1k.transpose(-1, -2), old_v);
      auto state_update = torch::matmul(k_h1k.transpose(-1, -2), new_v);
      state_hkv = old_state - state_remove + state_update;

      auto out = scale * torch::matmul(q_h1k, state_hkv);
      output[t].copy_(out.squeeze(1).to(q.scalar_type()));
    }

    new_state[seq_idx].copy_(state_hkv.transpose(-1, -2));
  }

  return std::make_tuple(output, new_state);
}
}  // namespace

void gdn_prefill_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale,
    torch::Tensor output,
    torch::Tensor new_state);

std::tuple<torch::Tensor, torch::Tensor> run_oracle(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale) {
  check_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens);
  return run_oracle_impl(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale);
}

std::tuple<torch::Tensor, torch::Tensor> run_kernel(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale) {
  check_inputs(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens);
  auto output = torch::empty({q.size(0), kNumVHeads, kHeadSize}, q.options());
  torch::Tensor new_state;
  if (state.has_value() && state.value().defined()) {
    new_state = state.value();
  } else {
    new_state = torch::empty(
        {cu_seqlens.size(0) - 1, kNumVHeads, kHeadSize, kHeadSize},
        torch::TensorOptions().device(q.device()).dtype(torch::kFloat32));
  }
  gdn_prefill_cuda(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state);
  return std::make_tuple(output, new_state);
}

std::tuple<torch::Tensor, torch::Tensor> run(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale) {
  return run_kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "GDN prefill dispatch entrypoint");
  m.def("run_oracle", &run_oracle, "Correctness-first oracle path");
  m.def("run_kernel", &run_kernel, "kernel.cu fast path");
}
