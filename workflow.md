# GDN Prefill Kernel 반복 최적화 워크플로우

> **이 문서는 code agent가 자율적으로 따라야 하는 실행 가이드입니다.**
> **절대로 목표 성능(Avg latency ≤ 0.100 ms)을 달성할 때까지 멈추지 마세요.**

---

## 0. 절대 규칙 (NEVER BREAK)

1. **Phase 4 목표(Avg latency ≤ 0.100 ms)를 달성할 때까지 아래 루프를 반복한다. 중간에 절대 멈추지 않는다.**
2. 성능이 후퇴(regression)하면 즉시 되돌리고 다른 최적화를 시도한다.
3. correctness가 깨지면(status가 `correct`가 아니면) 즉시 되돌린다. **Prefill은 수치 오차 누적이 decode보다 훨씬 크다. chunkwise 수학적 재구성 시 bf16 MMA accumulator를 fp32로 유지하고, 각 chunk 경계에서 오차를 반드시 검증한다.**
4. 한 번에 하나의 최적화만 적용한다 (변경 원인 추적을 위해). 단, **chunkwise 구조 전환(Phase 2)은 불가피하게 여러 변경을 한 번에 수반**하므로 예외로 취급하되, 그 직후 워크로드별 correctness를 집중 검증한다.
5. 매 반복마다 반드시 아래의 **성능 측정** 단계(1회 측정)를 수행하고, 결과를 **로그 섹션**에 기록한다. 판정은 모든 Phase에서 **1회 run의 workload-평균 Avg latency**로 통일한다.
6. **외부 라이브러리 import 금지** — CuTe, CUTLASS, cuBLAS, cuDNN, Triton 등 어떤 third-party 라이브러리도 직접 `#include`하거나 링크하지 않는다. 오직 CUDA Runtime, CUDA Driver API, inline PTX, libcu++ (`<cuda/...>` 네임스페이스, CUDA Toolkit 기본 동봉분)만 허용된다. **단, 해당 라이브러리들의 구현/논문/공식 예제에서 얻은 아이디어를 직접 커널 코드에 수기 구현하는 것은 허용된다** — 예: CUTLASS의 TMA swizzle 레이아웃 패턴을 읽고 같은 수학적 swizzle을 직접 inline PTX로 작성. 어떤 라이브러리의 어떤 기법을 참조했는지 iteration 로그에 명시한다.
7. **구조적 변경(chunkwise, WY, Tensor Core 도입) 전에는 항상 현재 kernel.cu 전체를 백업(기억)해둔다.** 이 변경들은 실패 시 점진 롤백이 어렵다.

---

## 1. 목표 정의

| Phase | 목표 Avg Latency | 핵심 전략 | 판정 기준 | 현재 상태 |
|-------|-----------------|----------|----------|----------|
| Phase 0 | — | 프로파일링 / 병목 확인 | 1회 측정 | **시작점 (0.178 ms)** |
| Phase 1 | ≤ 0.150 ms | 현재 recurrent 커널 baseline 튜닝 | **1회 측정 Avg** | 진행 중 |
| Phase 2 | ≤ 0.125 ms | **Chunkwise parallel 구조로 전환** | **1회 측정 Avg** | 진행 중 |
| Phase 3 | ≤ 0.110 ms | **Tensor Core (wgmma / tcgen05) + TMA** | **1회 측정 Avg** | 진행 중 |
| Phase 4 | ≤ 0.100 ms | Async pipeline + warp specialization + cluster | **1회 측정 Avg** | 진행 중 |

- **시작 성능**: Avg latency = 0.178 ms (recurrent, non-tensor-core 구현)
- **최대 기대 효과 구간**: Phase 2 (chunkwise 전환). 이유: 현재 커널은 T개 토큰을 완전 순차 처리 → O(T) critical path. Chunk size C로 나누면 O(T/C) sequential + O(C) parallel GEMM으로 변환되어 B200의 FLOPS/bytes 비율을 비로소 활용.
- **Phase 3은 Blackwell 전용 기능의 본격 도입 구간**: bf16 입력을 fp32 accumulator로 받는 wgmma/tcgen05 명령어, TMA bulk load 등.
- **Phase 4는 한 자리수 μs 단위 튜닝**: decode와 달리 prefill은 kernel 내부 시간이 여전히 > 100 μs이므로 launch overhead 문제는 상대적으로 작다. 대신 **compute/memory overlap 효율**과 **tail effect**가 핵심.

---

## 2. 파일 위치

| 항목 | 경로 |
|------|------|
| **커널 소스** | `solution/cuda/kernel.cu` |
| **패킹 스크립트** | `scripts/pack_solution.py` |
| **벤치마크 실행** | `scripts/run_modal.py` |
| **설정** | `config.toml` |
| **참조 Triton 구현** | `fla-org/flash-linear-attention`의 `gated_delta_rule` (chunkwise_fwd) |

---

## 3. 성능 측정 방법

매 반복의 측정 단계에서 **반드시** 아래 두 명령을 순차적으로 실행한다:

```bash
python scripts/pack_solution.py
modal run scripts/run_modal.py
```

출력에서 다음을 확인한다:
- 각 workload의 `status` → 반드시 모두 `correct`여야 함
- **`Avg latency: X.XXX ms`** → 이 값(모든 workload의 산술 평균)이 목표 이하인지 확인
- workload별 latency 분포도 함께 기록 (prefill은 seq_len에 따라 latency가 선형 증가하므로, 특정 workload에서만 후퇴하는 패턴 포착 중요)

**판정은 1회 측정의 Avg latency로 한다.** Modal 노이즈로 인해 경계값(예: 목표치 ±0.003 ms)인 경우에만 선택적으로 1~2회 재측정하여 확인한다. 평시에는 1회 run으로 유지/롤백 결정.

**추가 profiling (10회 반복 이상 답보 시)**:
```bash
# Modal container 내에서
ncu --set full --kernel-name gdn_prefill_kernel -o profile.ncu-rep ./run_once
# Or SASS 분석
cuobjdump --dump-sass kernel.cubin | grep -E "LDG|STG|HMMA|WGMMA|TMA"
```
병목 단서(memory-bound vs compute-bound, warp stall reason)를 찾아 다음 전략 선택에 반영.

---

## 4. 반복 루프 (매 iteration마다 수행)

```
┌─────────────────────────────────────────────────────┐
│  STEP 1: 현재 상태 확인                               │
│  - 직전 Avg latency 값을 확인한다                      │
│  - 목표 달성 여부를 판단한다                            │
│  - 달성했으면 → 다음 Phase로 / Phase 4 달성이면 종료    │
├─────────────────────────────────────────────────────┤
│  STEP 2: 최적화 전략 선택                              │
│  - 아래 "최적화 후보 목록"에서 아직 시도하지 않은 것 선택  │
│  - 현재 Phase에 맞는 카테고리 우선 (아래 참조)           │
│  - 예상 효과와 구현 난이도를 간단히 분석                  │
│  - 구현 계획을 문장으로 정리                           │
├─────────────────────────────────────────────────────┤
│  STEP 3: 커널 수정                                    │
│  - solution/cuda/kernel.cu 를 수정한다                 │
│  - 수정 전 현재 커널 전체를 백업(기억)해둔다              │
│  - 대규모 구조 변경(chunkwise, Tensor Core 도입) 시      │
│    먼저 pseudo-code로 설계 검증 후 구현                │
├─────────────────────────────────────────────────────┤
│  STEP 4: 성능 측정                                    │
│  - python scripts/pack_solution.py 실행               │
│  - modal run scripts/run_modal.py 실행 (1회)           │
│  - workload 평균 Avg latency와 status를 기록           │
│  - 경계값이면 선택적으로 1~2회 추가 측정                 │
├─────────────────────────────────────────────────────┤
│  STEP 5: 결과 판정                                    │
│  - correctness 실패 → 즉시 롤백, STEP 2로             │
│  - latency 후퇴 → 즉시 롤백, STEP 2로                │
│  - latency 개선 → 변경 유지, STEP 1로                 │
│  - latency 동일 → 유지/롤백 판단 후 STEP 2로          │
└─────────────────────────────────────────────────────┘
```

**Phase별 우선 카테고리:**
- Phase 1 (baseline 튜닝): 카테고리 E, F, H, K 우선
- Phase 2 (chunkwise 전환): 카테고리 A, B, **M** 우선 ← **가장 큰 한 방**
- Phase 3 (Tensor Core + TMA): 카테고리 C, D 우선
- Phase 4 (돌파구): 카테고리 I, J 우선

**이 루프를 Phase 4 목표(≤ 0.100 ms) 달성까지 반복한다. 절대 중단하지 않는다.**

---

## 5. 타겟 하드웨어: NVIDIA B200 (Blackwell, sm_100a)

이 커널은 **NVIDIA B200 GPU**에서 벤치마크된다. 최적화 시 아래 스펙을 반드시 참고한다.

| 항목 | 수치 | Prefill 관점의 시사점 |
|------|------|---------------------|
| Compute Capability | **10.0 (sm_100 / sm_100a)** | `sm_100a`로 컴파일해야 tcgen05, TMA bulk tensor 사용 가능 |
| Peak bf16 MMA TFLOPS | **~4.5 PFLOPS (sparse, per B200)** | Prefill은 compute-bound 구간 존재 → Tensor Core 필수 |
| Peak fp32 FMA | **~80 TFLOPS** | Tensor Core 없이 dot product는 FLOPS/bytes 비율이 매우 낮음 |
| L2 캐시 | **126 MB** | 모든 batch에서 state(4~64MB)가 L2에 상주. K/V 시퀀스도 대부분 L2 fit |
| Shared Memory/SM | **228 KB** (블록당 최대 227 KB) | Chunkwise에서 K/V chunk + WY tile 동시 보관 가능 |
| Max Warps/SM | **64** | Chunkwise는 큰 block이 유리 → 4~8 warps/block × 8 blocks/SM 정도 |
| Max Thread Blocks/SM | **32** | 큰 block 적게 돌리는 편이 Tensor Core 활용에 유리 |
| Register File/SM | **64K × 32-bit** | Prefill은 accumulator가 많아 register pressure 주의 (WY: 128×64 fp32 = 32KB) |
| HBM3e Bandwidth | **~8 TB/s** | Q/K/V bf16 stream이 크므로 TMA로 대역 포화 필요 |
| Thread Block Clusters | **최대 16 블록** (nonportable) | V_PER_Q=2이므로 2-CTA cluster로 q/k 공유 가능 |
| Distributed Shared Memory | 지원 | Cluster 내 q/k broadcast에 유용 |
| **TMA (Tensor Memory Accelerator)** | 지원 | **Prefill에서 매우 중요**: Q/K/V chunk를 `cp.async.bulk.tensor.2d`로 bulk load |
| **TMEM (Tensor Memory)** | **tcgen05 전용** | **Prefill에서 매우 중요**: B200의 5세대 tensor core, bf16→fp32 누산 |
| `wgmma.mma_async` (Hopper 호환) | 지원 | sm_100에서도 사용 가능. 구현 난이도 낮은 쪽 |
| `tcgen05.mma` (B200 전용) | 지원 | 더 높은 throughput, TMEM 필요. 구현 난이도 높음 |
| Kernel Launch Overhead | ~2-5μs | **100μs 타겟에서는 2~5%로 상대적으로 작음** — launch 최적화는 후순위 |

### Prefill 워크로드 크기 분석

```
가정: B ∈ {1, 2, 4, 8, 16}, T ∈ {512, 1024, 2048, 4096} (workload별 상이)

토큰당 bytes:
  Q: 4 × 128 × 2B = 1 KB
  K: 4 × 128 × 2B = 1 KB  
  V: 8 × 128 × 2B = 2 KB
  a, b: 8 × 2B × 2 = 32 B
  Output: 8 × 128 × 2B = 2 KB
  총 streaming: ~6 KB/token

State per (batch, v_head) = 128 × 128 × 4B = 64 KB
State per batch            = 8 × 64 KB     = 512 KB
B=16 total state           = 8 MB           → L2 126MB의 6.3% (완전 상주)

Total tokens 예: B=1,T=4096 → 4096 tokens → ~24 MB streaming input
  B=16,T=1024 → 16384 tokens → ~96 MB streaming input → L2에 부분만 fit
```

**결론: Prefill은 K/V 시퀀스 streaming이 지배적 bandwidth 소비자. State는 항상 L2 상주 가능. 
→ TMA로 K/V chunk prefetch + L2 persistence(state)의 조합이 핵심.**

### 산술 강도 (Arithmetic Intensity) 분석

**현재 recurrent 커널**:
```
per token (per v_head, per row):
  - Load k_vec (128 bf16 = 256B streaming)
  - Load state row (128 fp32 = 512B, L2 hit)
  - Compute: 4 FMA × 32 lanes + broadcast + 4 FMA × 32 lanes = ~256 FLOP
  - Store state row (128 fp32 = 512B)
  
Arithmetic intensity ≈ 256 FLOP / 1280 B ≈ 0.2 FLOP/B → severely memory-bound
```

**Chunkwise (C=64)**:
```
per chunk (per v_head, per row):
  - Load K chunk (64 × 128 bf16 = 16 KB), V chunk (64 × 128 bf16 = 16 KB) — once
  - State read/write (64 KB) — once per chunk, not per token
  - Compute: C² × D + C × D² = 64²×128 + 64×128² ≈ 1.5 M FLOP (WY + state update + output)
  - Store output chunk (64 × 128 bf16 = 16 KB)
  
Arithmetic intensity ≈ 1.5 M FLOP / 64 KB ≈ 24 FLOP/B → Tensor Core 영역
```

**→ Chunkwise 전환만으로 arithmetic intensity가 100배 가까이 상승. 이것이 Phase 2의 이론적 근거.**

---

## 6. 외부 라이브러리 정책 및 참고 구현

### 6.1. 외부 라이브러리 사용 정책 (엄수)

**허용:**
- CUDA Runtime API (`<cuda_runtime.h>`), CUDA Driver API (`<cuda.h>`)
- Inline PTX (`asm volatile(...)`) — Blackwell 전용 명령어 직접 사용 가능
- libcu++ (`<cuda/std/...>`, `<cuda/pipeline>`, `<cuda/barrier>`, `<cuda/annotated_ptr>`, `<cuda/atomic>`) — CUDA Toolkit에 기본 동봉
- CUDA bf16/half 타입 헤더 (`<cuda_bf16.h>`, `<cuda_fp16.h>`)
- PyTorch 바인딩용 `<torch/extension.h>`, `<ATen/...>`, `<c10/...>` (기존 사용 유지)
- CUB/Thrust (CUDA Toolkit 기본 동봉분 한정, device-side) — 필요 시에만

**금지:**
- CUTLASS, CuTe 헤더 (`<cutlass/...>`, `<cute/...>`) 직접 `#include` — **절대 불가**
- cuBLAS, cuBLASLt, cuDNN 링크 — **절대 불가**
- Triton JIT 바인딩, FlashAttention/FLA 바인딩 — **절대 불가**
- 외부 GitHub 저장소에서 복사한 include 경로, pre-compiled `.so`/`.a` 링크 — **절대 불가**

**회색 지대 (허용되지만 명시 필요):**
- 외부 프로젝트의 오픈소스 코드(아이디어/수학 구조/레이아웃 패턴)를 학습 후 **같은 로직을 직접 작성**하여 커널에 포함 — **허용**. 단, iteration 로그에 **"참조: {프로젝트명} / {파일}:{함수}"** 형태로 출처 명시.
- 외부 코드를 그대로 복사해 이름만 바꾼 경우는 **복사로 간주하여 금지**. 반드시 본인이 해당 GDN 컨텍스트(bf16, D=128, 8 v_heads, GQA=2, cu_seqlens 등)에 맞게 재설계해야 한다.

**이 정책의 이유:** 벤치마크 제출 환경은 단일 `kernel.cu` 소스와 `config.toml`만 전달된다. 외부 라이브러리를 가정하면 재현 불가.

### 6.2. 참고할 오픈소스 구현 (아이디어 추출용)

아래 프로젝트들은 **코드를 읽고 아이디어/패턴을 익힌 뒤 직접 구현**하는 데 쓴다. `#include`하지 않는다.

| 프로젝트 | 파일/모듈 | 추출할 아이디어 |
|---------|---------|--------------|
| **FLA (flash-linear-attention)** | `fla/ops/gated_delta_rule/chunk.py` | Chunkwise 수학 구조 그대로 포팅 (GDN의 WY + state 전이 + inter/intra output). 가장 직접적인 reference. |
| **FLA** | `fla/ops/gated_delta_rule/wy_fast.py` | WY representation의 triangular solve 구조, `fwd_prepare_wy_repr_kernel_chunk64` 패턴 |
| **CUTLASS (C++ 코드만)** | `include/cutlass/arch/mma_sm100.h`, `include/cute/atom/mma_traits_sm100*.h` | `tcgen05.mma` / `wgmma.mma_async` PTX 인스트럭션 시그니처, fragment layout, TMEM 주소 계산 |
| **CUTLASS** | `include/cute/atom/copy_traits_sm100*.h`, `sm90_tma_copy_*.hpp` | TMA descriptor 생성, `cp.async.bulk.tensor.2d/5d` 호출 시퀀스, multicast 패턴 |
| **CUTLASS** | `include/cute/swizzle.hpp`, `include/cute/swizzle_layout.hpp` | SMEM swizzle 수학(Sw<3,3,3> 등) — 수식을 읽고 같은 XOR 패턴을 직접 구현 |
| **CUTLASS examples** | `examples/49_hopper_gemm_with_collective_builder`, `examples/70_blackwell_gemm` | Warp-specialized mainloop 구조, producer/consumer 역할 분담 패턴 |
| **FlashAttention v3** | `csrc/flash_attn_hopper/flash_fwd_kernel.h` | Async pipeline + warp specialization + multi-stage `mbarrier` 사용 패턴. GDN prefill 구조와 가장 유사. |
| **Mamba / Mamba2** | `mamba_ssm/ops/triton/ssd_chunk_scan.py`, `ssd_state_passing.py` | Chunkwise state passing의 workload balancing, persistent block 스케줄링 |
| **ThunderKittens** | `kittens/ops/...` | Tile 추상화 아이디어 (단, 헤더를 포함하지 않고 인라인으로 유사한 wrapper 직접 작성) |
| **CCCL / cub** | `cub/block/block_scan.cuh` | Kogge-Stone / Brent-Kung scan 패턴 (CUB는 CUDA Toolkit 동봉이라 직접 사용 가능하지만, 핫패스에서는 수기 구현이 더 빠를 수 있음) |
| **NVIDIA cutlass_archtags / CUTLASS profiler** | `tools/util/include/cutlass/util/device_memory.h` (등) | L2 persistence + stream attribute 설정 예시 |
| **NVIDIA open kernels** | `nvidia/open-gpu-kernel-modules` 내 `sm100` 관련 헤더 | SM100 전용 feature enable 플래그, architecture-specific 상수 |

### 6.3. PTX / libcu++ 직접 사용 목록 (자주 쓰게 될 것들)

외부 라이브러리 대체 자주 쓸 저수준 프리미티브:

```
TMA bulk copy:           cp.async.bulk.tensor.2d.shared::cluster.global [smem], [tma_desc], {coords};
TMA bulk copy async완료: mbarrier.arrive.expect_tx / mbarrier.try_wait / mbarrier.wait
ldmatrix:                ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
wgmma.mma_async (SM90+): wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 ...
tcgen05.mma   (SM100+):  tcgen05.mma.cta_group::1.kind::f16 [tmem], [smem_a], [smem_b], ...
TMEM alloc/dealloc:      tcgen05.alloc / tcgen05.dealloc / tcgen05.commit / tcgen05.relinquish_alloc_permit
Async barrier:           mbarrier.init.shared / mbarrier.inval.shared
Cluster launch:          __cluster_dims__(X,Y,Z) / cudaLaunchKernelEx with clusterDim attribute
Distributed SMEM:        cluster.mapa.shared::cluster / cp.async.bulk.tensor ... .shared::cluster
L2 persistence:          cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)
Annotated ptr:           cuda::annotated_ptr<T, cuda::access_property::persisting>
Pipeline (libcu++):      cuda::pipeline<cuda::thread_scope_block, N>, cuda::memcpy_async
```

위 프리미티브들은 모두 CUDA Toolkit/libcu++ 범주이므로 허용된다. CuTe/CUTLASS 없이도 충분히 wgmma, TMA, TMEM을 구현할 수 있다 — 다만 헬퍼를 직접 작성해야 한다.

### 6.4. 직접 구현해야 하는 CUTLASS/CuTe 수준 유틸

아래는 CuTe/CUTLASS를 가져오면 즉시 해결되지만, 금지 정책상 직접 짜야 하는 것들. 코드를 간단히 스케치 해두고 필요 시점에 구현:

- **TMA descriptor builder**: host에서 `cuTensorMapEncodeTiled` 직접 호출. Q/K/V 각각 1회 생성 후 커널에 전달.
- **SMEM swizzle layout wrapper**: CuTe의 `Swizzle<3,3,3>`를 index 변환 함수로 직접 작성 (`smem_idx = base_idx ^ (((base_idx >> 3) & 7) << 3)` 등).
- **Tile iterator**: (batch, v_head, chunk) 좌표를 grid index에서 역산하는 작은 struct.
- **Persistent block scheduler**: global atomic counter로 work ticket을 pull하는 while loop.
- **Warp-group MMA wrapper**: `wgmma.mma_async`를 여러 shape/dtype에 대해 PTX-level 템플릿으로 묶기.
- **Async pipeline state machine**: `cuda::pipeline<>`을 쓰거나, 직접 mbarrier + stage index로 작성.

---

## 7. 최적화 후보 목록

아래는 시도할 수 있는 최적화 방향이다. **Phase 1 → 2 → 3 → 4 순서로 굵직한 구조 변경**을 거치는 것이 원칙이며, 각 카테고리 내에서는 구현 난이도가 낮은 것부터 시도한다.
시도한 것은 `[시도됨]` 표시를 하고 결과를 기록한다.

각 카테고리의 "**참조**" 줄은 아이디어를 추출할 오픈소스 위치를 가리킨다 (직접 import 아님).

---

### A. Chunkwise Parallel 구조 재설계 (★ Phase 2 핵심 ★)

**현재 커널의 가장 큰 문제는 T개 토큰을 완전 순차 처리한다는 점이다. Gated DeltaNet의 chunkwise parallel form으로 재설계해야 한다.**

**참조**: FLA `fla/ops/gated_delta_rule/chunk.py` (수학 구조 + chunk loop), Mamba2 `ssd_chunk_scan.py` (workload balancing). **import 금지, 아이디어만 이식.**

수학적 재구성 (FLA Triton reference `chunk_gated_delta_rule_fwd` 참조):
```
주어진 chunk 경계: 0 = t_0 < t_1 < ... < t_M = T, chunk size C
chunk c에서:
  γ_c = diag(gate_t_c, gate_{t_c+1}, ..., gate_{t_{c+1}-1})   -- 누적 gating
  K_c, V_c, Q_c, β_c  (chunk 내 블록)
  
  U_c = (I - tri(β_c K_c K_c^T))^{-1} · β_c · V_c_modified    -- WY solve
  S_{c+1} = γ_c · S_c + K_c^T · U_c                          -- state 전이
  O_c = Q_c · S_c + tril(Q_c · K_c^T) ⊙ γ_intra · U_c        -- output (inter + intra)
```

- [ ] **A1. Chunk 단위 tile scheduling 도입**: Grid를 `(num_v_heads, num_chunks_total)`로 재구성. 단, chunk 간 state 의존성은 sequential → 한 (batch, v_head)의 chunk들은 **하나의 block이 순차 처리**. 서로 다른 (batch, v_head)는 병렬. 이것만으로도 sequential critical path를 `T → T/C`로 단축.
- [ ] **A2. 병렬로 계산할 수 있는 행렬 합치기 (GEMM fusion)**: 한 chunk 내에서:
  - `P = Q_c · K_c^T` (C×C), `Y = Q_c · S_c` (C×D), `Z = K_c^T · U_c` (D×D) 
  - 이 중 입력을 공유하는 연산을 하나의 wgmma 호출로 병합. 예: `[Q_c | K_c] · [K_c^T, ...]` 형태로 묶을 수 있는 곳.
  - 또한 8개 v_head 중 같은 q_head를 공유하는 2개씩 (V_PER_Q=2)은 `Q_c`를 1회만 load하여 공유.
- [ ] **A3. Chunk size 자동 조절 (seq_len + GQA 기반)**: workload별 최적 C 선택.
  - 짧은 seq (T ≤ 128): C=T (chunkwise 불필요, 단일 chunk)
  - 중간 seq (128 < T ≤ 1024): C=64
  - 긴 seq (T > 1024): C=128 (WY tile과 tensor core shape 배수 맞춤)
  - 컴파일 타임 템플릿화: `template<int CHUNK_SIZE> __global__ void gdn_prefill_kernel`
  - host에서 seq_len/GQA ratio 보고 dispatch.
- [ ] **A4. cu_seqlens 기반 동적 workload balancing**: 가변 seq_len 때문에 단순 grid(num_seqs, ...) 할당은 tail effect 심함. persistent-block + work-stealing queue로 chunk 단위 동적 할당.
  - Global atomic counter로 chunk ticket 발급. 각 block이 `atomicAdd(&chunk_queue, 1)`로 작업 획득.
  - 단, (batch, v_head) 내 chunk는 순차 의존성 있으므로 work queue는 **(batch, v_head) 단위**.
- [ ] **A5. 초기 chunk 경계 reduction 병렬화**: `γ_c = ∏ gate_t` 같은 누적곱은 chunk 내부는 merge-sort 스타일 prefix-product로 `O(log C)` 단계에 계산. shared memory 사용.

---

### B. WY Representation & Sequential 연산 수학적 최적화 (Phase 2 핵심)

Delta Rule의 `(I - tri(β K K^T))^{-1}`는 명시적 역행렬이 아닌 WY 표현으로 효율 계산.

**참조**: FLA `fla/ops/gated_delta_rule/wy_fast.py`의 `fwd_prepare_wy_repr_kernel_chunk64` (rank-1 update 순서), CUB `cub/block/block_scan.cuh` (prefix-sum 패턴). **수기 이식.**

- [ ] **B1. WY compact form 구현**: `W, Y` 행렬로 update를 누적 표현. FLA Triton의 `fwd_prepare_wy_repr_kernel_chunk32/64` 참조. Triangular solve를 `C` 단계의 rank-1 update로 분해.
  - Pseudo: 
    ```
    A[i,:] = β_i · K_i^T
    for j = 0..i-1:
      A[i,:] -= (A[i,:] · K_j) · A[j,:]  -- B2 참조
    W[i,:] = A[i,:]
    ```
  - 이 triangular solve는 C=64일 때 64단계 순차지만, 각 단계 내에서는 D=128 차원 병렬.
- [ ] **B2. Triangular solve를 merge-sort 스타일로 분해**: `O(C)` 순차를 `O(log C)` 단계로. Cyclic reduction 또는 Brent-Kung. C=64 기준 6단계. 각 단계에서 tensor core GEMM 수행 가능.
- [ ] **B3. `γ_c` 누적 gating을 log space에서 prefix-sum**: `log(γ) = cumsum(log(gate))` 후 `exp`. prefix-sum은 Kogge-Stone 또는 Blelloch scan으로 `O(log C)` 단계. shared memory.
- [ ] **B4. Inter/intra output을 하나의 fused update로 병합**: `O = Q S + tril(P) · U` 에서 `P = Q K^T`를 두 번 계산하지 않도록. 동일 wgmma 출력을 두 경로에 재사용 (register).
- [ ] **B5. State 전이 `S_{c+1} = diag(γ) · S_c + K^T · U`를 단일 GEMM fused**: `S_c`를 `diag(γ)`로 pre-scale한 후 `K^T · U`와 더하는 것 → `gmma` accumulator reset 대신 이전 값을 초기 accumulator로 사용 (`wgmma.mma_async` accumulator loading).

---

### C. Tensor Core 활용 (★ Phase 3 핵심 ★)

**현재 커널은 Tensor Core를 전혀 사용하지 않는다.** B200의 bf16 peak은 fp32 FMA의 ~50배 → 반드시 도입.

**참조**: CUTLASS `include/cutlass/arch/mma_sm100.h` 및 `cute/atom/mma_traits_sm100*.h`의 PTX 시그니처/fragment 레이아웃, FlashAttention v3 `csrc/flash_attn_hopper/flash_fwd_kernel.h`의 wgmma 호출 시퀀스. **PTX는 직접 `asm volatile`로 작성.**

- [ ] **C1. `wgmma.mma_async` (Hopper-compatible) 도입 먼저**: 구현 난이도 낮음. bf16 input, fp32 accumulator. shape `m64n128k16` 또는 `m64n64k16`.
  - 적용 대상: `Q·K^T` (C×C), `Q·S` (C×D), `K^T·U` (D×D), `P·U` (C×D)
  - 각 chunk 내 모든 주요 행렬곱을 wgmma로 교체.
- [ ] **C2. `ldmatrix` / `ldmatrix.trans`로 shared memory → register 로드**: wgmma operand A는 register 분산 로드 필요. `ldmatrix.x4.m8n8.shared.b16`로 16×16 tile을 4 threads에 분산.
- [ ] **C3. `tcgen05.mma` (B200 전용) 도입**: TMEM 기반 5세대 tensor core. wgmma 대비 throughput 향상. 단, TMEM 할당·해제(`tcgen05.alloc`, `tcgen05.dealloc`) 관리 필요. `sm_100a` 타겟 필수.
  - 우선순위: C1 → C2 → C3 순. C3는 Phase 4로 미룰 수도 있음.
- [ ] **C4. bf16 누산 (실험적)**: 표준은 fp32 accumulator지만, 일부 중간 행렬(P = QK^T)은 bf16으로도 허용되는지 correctness 실험. 메모리 절반으로 occupancy 향상 가능. **수치 안정성 주의**.
- [ ] **C5. Tensor Core shape에 맞춘 chunk size 강제**: `wgmma m64n128k16` 사용 시 C ∈ {64, 128}, D=128 (이미 M/N 배수). 미스매치 없이 완전 활용.

---

### D. TMA + Async Pipeline (Load-Compute Overlap, Phase 3 핵심)

**참조**: CUTLASS `cute/atom/copy_traits_sm100*.h` 및 `sm90_tma_copy_*.hpp` (TMA descriptor + multicast 패턴), FlashAttention v3 mainloop (3-stage pipeline 상태 전이), libcu++ `<cuda/pipeline>` 공식 예제. **`cuTensorMapEncodeTiled`는 Driver API라 직접 호출 허용.**

- [ ] **D1. TMA tensor map 생성 (host-side, 1회)**: `cuTensorMapEncodeTiled`로 Q/K/V tensor map 등록. `[total_seq_len, num_heads, head_dim]` 3D 텐서를 chunk 단위로 bulk load.
  - Host에서 3개 TensorMap (Q, K, V) 생성 후 커널 인자로 전달. 재사용.
- [ ] **D2. `cp.async.bulk.tensor.2d.shared::cluster.global`로 chunk bulk load**: 16 KB chunk를 한 번의 TMA instruction으로. coalescing, address 계산 하드웨어 처리.
- [ ] **D3. 3-stage async pipeline (load / compute / store)**:
  ```
  stage 0: compute chunk c   (data in SMEM tile 0)
  stage 1: TMA load chunk c+1 (filling SMEM tile 1)  -- in flight
  stage 2: issue TMA for c+2 (starting SMEM tile 2)
  ```
  `cuda::pipeline<thread_scope_block, 3>` + `cuda::memcpy_async` + `cuda::aligned_size_t<16>`.
- [ ] **D4. `mbarrier` 기반 async barrier**: TMA 완료를 `mbarrier.try_wait`로 polling. `__syncthreads` 대체. warp 단위 대기.
- [ ] **D5. State row async pre-fetch**: state는 L2 상주이지만 초기 chunk에서는 cold. L2 hit 보장 후에도 SMEM 사본으로 bring-down 가치 있음.
- [ ] **D6. Output write async**: `cp.async.bulk.tensor` 로 output chunk를 async store. compute ↔ store overlap.

---

### E. 메모리 접근 최적화 (Phase 1 / 전 단계)

**참조**: CUTLASS `include/cute/swizzle.hpp` (Sw<3,3,3> XOR 패턴을 수식으로 이해 후 직접 구현), NVIDIA CUDA Programming Guide의 "L2 Persistence" 섹션, FlashAttention v2/v3의 SMEM swizzle 구현.

- [ ] **E1. bf16 vectorized load를 `__nv_bfloat162` packed로 교체**: 현재 `load_bf16x4`는 2 × `__nv_bfloat162` 로드이지만, 주소 정렬 및 transaction 수 재확인. 128-bit load로 통합 가능한지 `ldg.v8.b16` PTX 실험.
- [ ] **E2. State read/write를 `float4` vectorized로 통일** [부분 적용됨]: 이미 `reinterpret_cast<float4*>` 사용. 추가로 `ld.global.nc.v4.f32`로 non-coherent cache 명시.
- [ ] **E3. L2 Persistence + hitRatio 튜닝 (`cudaAccessPolicyWindow`)**: host 측에서 state 버퍼에 persistence hint. `hitRatio = min(persistingL2CacheMaxSize / total_state_bytes, 1.0f)` 공식 엄수. 또한 `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize)`. `hitProp=cudaAccessPropertyPersisting`, `missProp=cudaAccessPropertyStreaming`.
- [ ] **E4. Q/K/V를 streaming cache path로 (`__ldcs` / `cudaAccessPropertyStreaming`)**: K/V는 한 chunk에서 한 번만 read → L2를 오염시키지 않도록 streaming hint.
- [ ] **E5. Output은 `__stwt` (streaming store)**: write-through로 L2 오염 방지. **주의**: 이전 decode에서 후퇴한 이력 있으나 prefill은 output volume이 훨씬 커서 효과 다를 수 있음. 재실험 가치.
- [ ] **E6. Shared memory carveout 조절**: `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, <%>)`. Chunkwise는 SMEM heavy → 100% SMEM (228KB) 설정 필요.
- [ ] **E7. Bank conflict 정밀 분석**: K/V tile이 SMEM에 올라간 후 `ldmatrix` access 패턴이 bank-conflict-free인지 확인. Swizzle 적용 (`cp.async.bulk`의 swizzle mode 사용 또는 수동 `^`-swizzling).

---

### F. 레지스터 caching & Double Buffering (Phase 1 / 3)

**참조**: CUTLASS `gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` (register accumulator 분할), ThunderKittens `kittens/ops/...`의 tile 추상화 (헤더 포함 금지, 아이디어만).

- [ ] **F1. Q chunk를 register tile로 유지**: `Q_c`는 chunk 내 여러 GEMM에 재사용 (`Q·K^T`, `Q·S`). register에 상주시키면 SMEM round-trip 제거. 단, register pressure 주의.
- [ ] **F2. State를 register로 prefetch (recurrent 경로 유지 시)**: 현재 state_vec는 이미 register. 확장: 2-row 또는 4-row 병렬 prefetch.
- [ ] **F3. Gate/beta 값을 register에 batch 로드**: 현재 매 토큰마다 `gate_beta[t * H + h]` 읽음. chunk 시작 시 C개를 한 번에 로드 → register 또는 SMEM.
- [ ] **F4. Double-buffered SMEM tiles for K/V**: ping-pong buffer 2개. 짝수 chunk는 buffer A, 홀수는 buffer B. wgmma issue 후 다음 chunk TMA 동시 진행.
- [ ] **F5. Accumulator register 재사용**: output accumulator와 state accumulator를 wgmma 출력 사이에 swap. register 총합 관리.

---

### G. Sync 최소화 / Kernel Fusion (Phase 1 / 3)

**참조**: NVIDIA PTX ISA "Parallel Synchronization and Communication Instructions" (mbarrier 시맨틱), FlashAttention v3의 mbarrier-based barrier 예제, CUB `cub/block/block_scan.cuh`.

- [ ] **G1. `compute_gate_beta_kernel`을 메인 커널에 fusion**: 현재 별도 launch 오버헤드 + global round-trip. 메인 커널 시작부에서 각 block이 자기 chunk의 gate/beta만 계산하고 register/SMEM 유지.
  - **주의**: gate/beta는 v_head 단위, 메인 커널은 (v_head, row) 단위. row 간 중복 계산 피해야 함 → warp 1개가 전담하고 나머지에 broadcast.
- [ ] **G2. `__syncthreads()` → `mbarrier` 기반 warp-level sync**: 전체 block sync를 warp-local arrive/wait로 대체 가능한 구간 탐색.
- [ ] **G3. `__syncwarp()` 불필요 제거**: warp-level 연산 중 `__shfl_*_sync`는 implicit sync이므로 명시적 `__syncwarp` 불필요.
- [ ] **G4. Chunk 경계 sync 최소화**: state 전이는 chunk 간 barrier 필요하지만, output write는 독립 → 별도 sync 불필요.
- [ ] **G5. PTX-level prefetch insertion**: `prefetch.global.L2 [addr];` 로 다음 chunk K/V 주소 prefetch. sync 없이 latency hiding.

---

### H. 실행 구성 최적화 (Phase 1)

**참조**: Mamba2 `ssd_chunk_scan.py` / CUTLASS persistent mainloop (`sm90_gemm_tma_warpspecialized_pingpong.hpp`)의 persistent block 스케줄링 패턴.

- [ ] **H1. Block size 재탐색**: 현재 2 warps = 64 threads. Chunkwise는 GEMM 단위가 커서 4 warps (128) 또는 8 warps (256)가 유리할 수 있음. wgmma는 warp-group(4 warps) 단위.
- [ ] **H2. `__launch_bounds__` 세밀 조정**: 현재 `(kThreads=64, 4)`. Chunkwise 도입 후 register 사용량 재측정 (`-Xptxas -v`). 예상: register 80~120/thread → occupancy 제약.
- [ ] **H3. Grid 재설계**: 현재 `(v_heads × row_tiles, num_seqs)` = (512, num_seqs). Chunkwise 전환 후 `(v_heads, num_seqs)` 또는 `(num_active_chunks)` (persistent 경우).
- [ ] **H4. Row tiling 제거 (chunkwise 이후)**: wgmma는 128×128 state 전체를 warp-group이 담당하므로, 현재의 row_tile 2-row 분할이 불필요해짐. Warp group 1개가 (v_head, chunk) 전체 처리.
- [ ] **H5. SM 수 기반 persistent kernel**: B200 SM ≈ 148. persistent block 148×2=296개 생성 후 chunk queue에서 work pull. launch overhead + tail effect 동시 해소.

---

### I. Warp Specialization (★ Phase 4 핵심 ★)

**참조**: CUTLASS `examples/70_blackwell_gemm` 및 `gemm/kernel/sm100_gemm_*.hpp`의 warp-specialized mainloop, FlashAttention v3 `flash_fwd_kernel.h` (producer/consumer role + `setmaxnreg`), NVIDIA blog "Accelerating GEMM on Hopper" (producer-consumer 분리 다이어그램).

- [ ] **I1. Producer/Consumer warp 분리**: 예) 8 warps/block 중 1 warp TMA 전담(producer), 나머지 7 warps는 wgmma 전담(consumer). `mbarrier`로 동기화.
  - `setmaxnreg.inc` / `setmaxnreg.dec` (Blackwell)로 warp role별 register 할당 다르게.
- [ ] **I2. Math warp + epilogue warp 분리**: GEMM 전담 warp과 softmax/epilogue 전담 warp 분리. 겹치는 chunk에서 동시 실행.
- [ ] **I3. `__nanosleep(0)` + mbarrier polling**: busy-wait 대신 energy-aware polling.

---

### J. 2-CTA Cluster / Distributed SMEM (Phase 3/4)

**참조**: CUTLASS `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 중 cluster_shape=(2,1,1) 구간, NVIDIA CUDA Programming Guide "Thread Block Clusters" + "Distributed Shared Memory" 섹션.

- [ ] **J1. 2-CTA cluster로 Q 공유 (V_PER_Q=2)**: 같은 q_head를 쓰는 2개 v_head block을 cluster로 묶어 Q chunk를 distributed SMEM으로 공유. Q bf16 read를 절반으로.
  - `__cluster_dims__(2, 1, 1)` 또는 `cudaLaunchKernelEx` + `cudaLaunchAttribute::clusterDim=(2,1,1)`.
- [ ] **J2. 4-CTA cluster로 Q+K 공유 (V_PER_K=2)**: V_PER_K=2이므로 4 v_head를 묶으면 K도 공유 가능. 단, cluster 크기가 커지면 synchronization overhead 증가.
- [ ] **J3. Distributed SMEM으로 partial output aggregation**: split 전략 쓸 때 cluster 내 partial output을 dsmem으로 모아 한 번에 global write.
- [ ] **J4. `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in**: cluster size 8/16 실험.

---

### K. 컴파일러 힌트 및 빌드 옵션 (Phase 1)

**참조**: NVCC 공식 문서 (`nvcc --help`의 `-gencode`, `-Xptxas`, `-maxrregcount` 섹션). 외부 라이브러리 불필요.

- [ ] **K1. `-arch=sm_100a` 명시**: Blackwell B200 전용 (TMA bulk tensor, tcgen05, fast wgmma) 활성화. `sm_100f` (family-specific) 대안 비교.
- [ ] **K2. `-Xptxas -v -Xptxas -warn-spills`로 register/spill 분석**: 현재 커널 register/spill 상태 파악. chunkwise 전환 후 반드시 재측정.
- [ ] **K3. `__builtin_assume()` 힌트**: `__builtin_assume(blockDim.x == 128)`, `__builtin_assume(num_seqs > 0)`, `__builtin_assume(seq_end > seq_start)` 등. 분기 제거.
- [ ] **K4. `__forceinline__` 모든 device helper**: 이미 적용됨. 신규 helper 추가 시 누락 주의.
- [ ] **K5. `-maxrregcount=N` 조합**: chunkwise 후 register spill 발생 시 backoff 수단.
- [ ] **K6. `#pragma unroll` 세밀 적용**: chunk 내 순차 루프 중 컴파일 타임 상수 것만 unroll. 너무 aggressive한 unroll은 I-cache miss.
- [ ] **K7. `-use_fast_math` 국소 적용 (주의)**: `__expf`, `__logf` 등 intrinsic은 gate/beta 계산에만 국소 적용. 전역은 correctness 위험.

---

### L. 안정성·분산 감소 (Phase 3/4)

**참조**: 자체 Modal 환경 측정 경험. 외부 라이브러리 불필요.

- [ ] **L1. L2 cache warmup kernel 선행 실행**: 벤치마크 측정 전 state를 L2에 올려둠. Modal 환경 noise 감소.
- [ ] **L2. Grid 크기를 SM 수(148) 배수에 맞춤**: tail effect 감소. 특히 짧은 seq batch에서 중요.
- [ ] **L3. Stream priority 최고**: `cudaStreamCreateWithPriority` 최고 우선순위.
- [ ] **L4. Warmup iter 증가**: `config.toml`의 warmup=10 → 20.
- [ ] **L5. `cudaDeviceSynchronize()` 불필요 호출 제거**: Modal bench script 확인.
- [ ] **L6. CUDA Graph capture (후순위, 100μs 타겟에서는 이득 작음)**: kernel launch overhead 2~5μs는 전체의 2~5%. decode와 달리 우선도 낮음. Phase 4 말미에 여유 있으면 시도.

---

### M. 수학적 / 알고리즘적 재구성 (★ Phase 2/3 보조 — 일부는 Phase 2 필수 ★)

**현재 커널의 연산을 수학적으로 등가이지만 더 저렴한 형태로 재구성. Tensor Core 여부와 무관하게 FLOP 자체를 줄이거나, 데이터 의존성을 완화하거나, 중간 결과를 vector로 축소하여 SMEM/register 압력을 낮춘다. Winograd convolution, prefix-sum LUT, associativity rewrite 같은 "compute-invariant but cheaper" 변환의 GDN 맥락 대응물.**

**참조**: FLA `chunk.py` / `wy_fast.py` (아래 트릭 일부가 이미 구현되어 있음 — 반드시 읽고 확인), Retention / RWKV / Mamba2 논문의 linear attention 수학, Numerical Recipes (Kahan summation, minimax polynomial 근사). **전부 수기 이식.**

- [ ] **M1. `γ_intra` 행렬 대신 Q/K scaling으로 인수분해 (★ 최우선, Phase 2 필수 ★)**:
  - 정의: `γ_intra[i,j] = ∏_{k=j+1}^{i} gate_k` (C×C 행렬, intra-chunk causal masking에 사용)
  - 핵심 identity: `γ_cum[i] = ∏_{k=0}^{i} gate_k`로 두면 `γ_intra[i,j] = γ_cum[i] / γ_cum[j]`
  - 따라서 intra output을 다음과 같이 재작성:
    ```
    O_intra[i] = Σ_{j≤i} Q[i] · K[j]^T · U[j] · γ_intra[i,j]
               = γ_cum[i] · Σ_{j≤i} Q[i] · (K[j]/γ_cum[j])^T · U[j]
               = γ_cum[i] · (tril(Q · K̃^T) · U)[i]    where K̃[j] = K[j]/γ_cum[j]
    ```
  - **효과**:
    - C×C `γ_intra` 행렬 생성/저장/elem-wise mul이 **완전히 제거** (C=64 기준 fp32 16 KB SMEM + C² FMA 삭제)
    - 대신 vector scaling 2회 (`K̃ = K ⊙ γ_cum_inv`, output `⊙ γ_cum`) — C×D FMA만 추가
  - **주의**: `γ_cum`이 매우 작거나 클 때 `1/γ_cum` 계산에 underflow/overflow 가능 → log-space 표현과 조합 권장 (M4와 연계)
  - Retention, RWKV, Mamba2에서 모두 사용하는 canonical 트릭. **반드시 채택.**

- [ ] **M2. GEMM 결합법칙 재배치 + rank-C deferred materialization (Phase 2 필수)**:
  - Intra output `(Q K^T) U` vs `Q (K^T U)`:
    | 배치 | 중간 shape | 중간 FLOP | 이후 FLOP | 총 FLOP |
    |------|-----------|----------|----------|---------|
    | `(Q K^T) U` | C×C | 2CCD | 2CCD | **4C²D** |
    | `Q (K^T U)` | D×D | 2CDD | 2CDD | **4CD²** |
  - C=64, D=128: `(Q K^T) U`가 4·64²·128 ≈ 2M, `Q (K^T U)`가 4·64·128² ≈ 4M → **전자가 2배 저렴**
  - 추가로 `(Q K^T)`는 intra-chunk causal mask `tril()`을 그대로 적용할 자연스러운 형태 → M1과 완벽히 호환
  - Inter output의 state contribution은 반대로 `Q · S_c` 하나뿐이므로 추가 GEMM 필요 없음
  - FLA `chunk_fwd_kernel_o`가 실제로 쓰는 배치. 이식 시 반드시 이 순서로.

- [ ] **M3. `γ_cum` / `γ_cum_inv`를 shared memory LUT로 precompute (★ prefix-sum LUT 패턴, Phase 2 필수 ★)**:
  - `γ_cum[0..C-1]`은 chunk 내 여러 경로에서 반복 사용 (M1의 K scaling, output post-scaling, 다음 chunk의 state scale, WY prepare의 gate weight 등)
  - 한 번 log-space prefix-sum으로 계산 (B3)하고 SMEM에 상주:
    ```
    gate_log[c] = log(gate_c)                    // C fp32
    gate_logcum[c] = prefix_sum(gate_log)        // C fp32, Kogge-Stone O(log C)
    gamma_cum[c] = exp(gate_logcum[c])           // C fp32
    gamma_cum_inv[c] = exp(-gate_logcum[c])      // C fp32
    ```
  - 저장량: 4 × C × 4B = 1 KB (C=64) — padding 포함해도 bank-conflict-free에 여유
  - 모든 consumer가 **단순 index load**로 재사용 → 재계산 완전 제거
  - **사용자 예시의 "prefix sum LUT"에 정확히 대응하는 구조**. 개별적으로 비용이 작아 보이지만, chunk당 수십 번 참조되므로 누적 효과 큼.

- [ ] **M4. 전 구간 log-space 유지로 overflow/underflow 방지 + 재계산 제거 (Phase 2 필수, M3과 연계)**:
  - `γ_cum`은 T가 길면 쉽게 `[1e-20, 1e+5]` 범위를 벗어남 (gate 분포에 따라)
  - M3의 LUT를 `gate_logcum` (log-space)만 유지하고, 각 consumer에서 필요 시 `exp`로 변환
  - 두 값을 빼는 연산(예: `γ_intra[i,j] = exp(gate_logcum[i] - gate_logcum[j])`)은 numerical stability 확보됨
  - 단, Tensor Core GEMM 입력 준비를 위해 `K̃ = K ⊙ exp(-gate_logcum)` bf16 텐서를 만들 때는 `exp`를 거쳐야 함 → overflow 체크 필요
  - M3 + M4는 함께 적용. **M4 없이 M3만 적용하면 긴 T에서 correctness 실패 가능.**

- [ ] **M5. `β K` 선(先)곱 적용 (trivial algebraic rewrite, free win, Phase 2 필수)**:
  - WY prepare의 `A[i,:] = β_i K_i^T`, state 전이의 `K^T · U` (여기서 U는 β가 이미 녹아든 값), intra output의 `K^T U` 모두 β_i · K_i 조합 사용
  - Chunk 시작 시 `K_scaled[i,:] = β_i · K[i,:]`를 한 번 계산해 SMEM 상주 → 모든 consumer가 공유
  - FLOP 절감: chunk당 C·D = 8192 bf16 mul 중복 제거 (C=64)
  - SMEM 추가: C·D·2B = 16 KB (단, 기존 K tile을 in-place overwrite 가능하면 0)
  - FLA `wy_fast.py`의 `fwd_prepare_wy_repr_kernel_chunk64`가 실제로 하는 rewriting.

- [ ] **M6. `K K^T`의 대칭성 활용 (WY prepare 국한)**:
  - `β K K^T`는 대칭 (`β_iβ_j K_i·K_j^T` 원소가 `(i,j)`와 `(j,i)`에서 동일)
  - Upper triangle만 계산 후 lower로 복사 → FLOP 약 50% 절감 (C=64 기준 C²·D/2 = 262K FLOP)
  - **제약**: wgmma / tcgen05은 symmetric 버전이 없어 hardware-level 활용 불가 → SIMT fallback 경로나 custom kernel에서만 유효
  - BLAS의 SYRK (`C = α A Aᵀ + β C`) 에 해당. 참조: CUTLASS의 `Sm90SyrK` 파일, 단 직접 PTX로 이식
  - Tensor Core 도입(Phase 3) 후에는 trade-off 재평가 필요. 우선순위 중간.

- [ ] **M7. softplus / sigmoid의 minimax 다항식 근사 (preprocessing 한정)**:
  - 현재 `compute_gate_beta_kernel`의 `softplus(a + dt_bias)`, `sigmoid(b)`: 각 `__expf` 호출 1~2회 = 10~20 cycle
  - 입력 범위가 대부분 bounded (`a + dt_bias ∈ [-8, 8]`가 학습 분포의 99.9% 점유) → degree-5 minimax 다항식으로 1~2 ULP 근사 가능 (~5 cycle)
  - 범위 밖은 fallback (클램핑 또는 원본 함수)
  - **Correctness 위험**: chunkwise 이후 gate가 T번 곱해지므로 ULP 오차가 exponential하게 증폭 → 반드시 먼저 T=4096에서 rtol/atol 테스트
  - 예상 효과: preprocess가 전체의 5~10% → 전체 1~3% 개선. 후순위.

- [ ] **M8. Intra output 계산 시 mask 저장 대신 causal index 분기 (SIMT 한정)**:
  - `tril(Q K̃^T)`의 bool mask를 register/SMEM에 materialize하지 않고, 각 thread가 담당 `(i,j)`의 `i ≥ j`만 FMA 수행, 나머지는 skip
  - Tensor Core 경로에서는 mask 자체가 compute 속에서 0 multiplicand로 처리되므로 이 최적화는 **SIMT 경로에서만** 유효
  - Phase 2 chunkwise 초기 SIMT 구현 시 적용, Phase 3 Tensor Core 전환 시 자연스럽게 사라짐

- [ ] **M9. State accumulation을 rank-C keep & late-materialize (실험적)**:
  - Chunk 하나가 state에 주는 기여 `ΔS_c = K_c^T · U_c`는 rank ≤ C (C=64 < D=128)
  - 이를 D×D로 materialize하지 않고 `(K_c^T, U_c)` pair로 유지 가능 — 다음 chunk의 output 계산 `Q · S_c = Q · (Σ prev ΔS_k)`에서 `(Q · K_k^T) · U_k`로 분해하면 `C×C → C×D` 두 개의 smaller GEMM
  - **제약**: 지난 chunks의 `(K_k^T, U_k)`를 모두 유지하려면 메모리가 O(num_chunks × C × D) — 감당 불가
  - **실용적 적용**: 직전 1~2 chunk만 rank-C 유지 (register/SMEM), 그 이전은 D×D로 materialize. 부분 최적화.
  - 효과 불확실, 구현 복잡. Phase 4 말미 실험 후순위.

- [ ] **M10. Compensated (Kahan) summation for long-T state accumulation (진단 도구)**:
  - 긴 T (≥ 4096) + 높은 gate (≈ 1) 조합에서 fp32 state accumulator 오차가 correctness threshold 근처에 도달 가능
  - Kahan pair `(S, c)` 유지 — FLOP 4× 증가하지만 오차 1 ULP로 제한
  - **chunkwise 전환하면 역설적으로 불필요**: chunk 경계마다 materialize → 오차 carry 단절
  - 진단 도구로만 준비해두고, 현재 correctness 통과하므로 기본 불적용.

**M 카테고리 우선순위 요약:**

| 트릭 | 효과 | 위험도 | 언제 |
|------|------|-------|------|
| **M1** (γ_intra 인수분해) | SMEM 16 KB + C² FMA 제거 | 낮음 (수학적 등가) | **Phase 2 필수** |
| **M2** (GEMM 결합 재배치) | FLOP 2× 절감 | 없음 | **Phase 2 필수** |
| **M3** (γ_cum SMEM LUT) | 재계산 제거, ~수십 회 재사용 | 없음 | **Phase 2 필수** |
| **M4** (log-space 유지) | long-T에서 correctness | 낮음 | **M3와 동시 적용** |
| **M5** (β K 선곱) | chunk당 8K FMA 제거 | 없음 | **Phase 2 필수** |
| M6 (대칭 KK^T) | WY prepare 50% 절감 | 중간 (TC 활용 불가) | Phase 2 말미 |
| M7 (minimax 근사) | preprocess 30% 단축 | 중간 (누적 오차) | Phase 4 |
| M8 (mask skip) | SIMT 경로만 유효 | 없음 | Phase 2 한정 |
| M9 (rank-C keep) | 불확실 | 높음 (구현 복잡) | Phase 4 |
| M10 (Kahan) | correctness 진단 | 없음 | 실패 시만 |

---

## 8. 현재 커널 구조 요약 (agent 참고용)

### 현재 (시작점, 0.178 ms)

```
Preproc kernel: compute_gate_beta_kernel
  - Grid: (ceil(total_seq_len * 8 / 256),)
  - Block: 256 threads, __launch_bounds__(256, 2)
  - 각 thread가 1개 (t, v_head) gate/beta 계산 → global memory write
  - a, b (bf16) + A_log, dt_bias (fp32) → gate_beta (fp32x2) 전처리
  - softplus(a + dt_bias), sigmoid(b)

Main kernel: gdn_prefill_kernel
  - Grid: (8 × 64 = 512, num_seqs, 1)  (v_heads × row_tiles, sequences)
  - Block: 64 threads (2 warps), __launch_bounds__(64, 4)
  - 각 block = 하나의 (seq, v_head, row_tile) 담당, 한 warp가 한 row 담당
  - 각 warp의 32 lane이 128-col을 4-float씩 분담 (state_vec = float4 per lane)
  - For t in [seq_start, seq_end):          -- SEQUENTIAL (병목!)
      q_vec, k_vec load (bf16→fp32x4)
      old_v = warp_sum(k_vec · state_vec) × gate       (scalar broadcast)
      diff = beta × (v - old_v)
      state_vec = state_vec × gate + k_vec × diff       (fp32 rank-1 update)
      out = warp_sum(q_vec · state_vec) × scale        (scalar, bf16 store)
  - state 최종값을 state_out에 float4 write

State: [num_seqs, 8, 128, 128] fp32 (k-last layout)
GQA: V_PER_Q = 2, V_PER_K = 2 (8 v_heads, 4 q_heads, 4 k_heads)
입력 bf16, state fp32, 출력 bf16
```

### B200에서의 리소스 사용 추정

```
Block size = 64 threads (2 warps), __launch_bounds__(64, 4):
- 레지스터/thread: 추정 ~40~50 (state_vec 4 + locals)
- Shared memory: 0 (현재 사용 안 함)
- 이론 occupancy: 4 blocks × 2 warps = 8 warps/SM = 12.5% (매우 낮음!)
- B200 SM ≈ 148개
- Grid: 512 × num_seqs → num_seqs=1일 때 512 blocks (SM당 ~3.5)
                      → num_seqs=16일 때 8192 blocks (충분)
```

### 현재 병목 분석

```
1. ★ SEQUENTIAL token loop (T iterations) — 가장 큰 병목 ★
   → chunkwise parallel로 전환 시 critical path T → T/C (C=64 기준 64배 단축)
2. ★ Tensor Core 완전 미활용 ★
   → bf16 peak의 수십 분의 1 성능
3. Occupancy 12.5% (2 warps/block × 4 blocks/SM)
   → 큰 block + 적은 block 전략으로 재설계 필요
4. compute_gate_beta_kernel 별도 launch + global round-trip
   → fusion 가능
5. Row tiling (kRowTilesPerHead=64, kRowsPerBlock=2)이 Tensor Core shape과 미스매치
   → warp group 1개가 128×128 state 전체 담당하도록 재설계
6. Scalar broadcast (old_v, v_val)이 매 토큰마다 warp_sum + shuffle
   → chunkwise에서는 한 번의 GEMM으로 일괄 처리 대체
```

---

## 9. 성능 로그

매 iteration마다 아래 형식으로 이 섹션에 추가 기록한다. **판정은 1회 측정의 workload 평균 Avg latency 기준**.

### Iteration 0 (시작점)
- 최적화: N/A (baseline)
- 변경 요약: 현재 recurrent prefill 커널
- Avg latency: **0.178 ms** (baseline, 1회 측정)
- Status: correct
- 판정: 시작점
- 현재 Phase: 0 → 1
- 관찰: Tensor Core 미사용, sequential token loop, occupancy 12.5%.

### Iteration N (템플릿)
```
### Iteration N
- 최적화: <카테고리 번호 + 간단 설명 (예: A1. chunkwise grid 재구성)>
- 변경 요약: <상세>
- 참조 구현: <있으면 "CUTLASS mma_sm100.h", "FLA wy_fast.py" 등. 없으면 "없음">
- Avg latency: X.XXX ms (이전: Y.YYY ms)
- 변화: ±X.XXX ms
- Status: correct / incorrect
- 판정: 유지 / 롤백
- 현재 Phase: N
- 인사이트: <이 iteration에서 얻은 교훈>
```

**보조 측정 (선택):** 결과가 목표 경계값(±0.003 ms)이면 1~2회 추가 측정한 값도 아래처럼 기록.
```
- 추가 측정: [X.XXX, X.XXX] ms
```

---

## 10. 완료 조건

- [ ] **Phase 1 달성**: Avg latency ≤ 0.150 ms (baseline 튜닝)
- [ ] **Phase 2 달성**: Avg latency ≤ 0.125 ms (chunkwise 전환)
- [ ] **Phase 3 달성**: Avg latency ≤ 0.110 ms (Tensor Core + TMA)
- [ ] **Phase 4 달성**: Avg latency ≤ 0.100 ms (async pipeline + cluster)

**Phase 4를 달성하면 이 워크플로우를 종료하고, 최종 결과를 사용자에게 보고한다.**
**Phase 4를 달성하지 못했으면 절대 멈추지 말고 루프를 계속 반복한다.**

---

## 11. 롤백 정책

- 커널 수정 전, 항상 현재 동작하는 **전체 커널 코드를 기억**해둔다.
- correctness 실패 또는 latency 후퇴 시, 즉시 직전의 정상 버전으로 `solution/cuda/kernel.cu`를 복원한다.
- 롤백 후 다른 최적화를 선택하여 다시 시도한다.
- 같은 최적화를 두 번 이상 실패했으면 해당 항목을 `[실패]` 처리하고 넘어간다.
- **대규모 구조 변경(chunkwise, Tensor Core 도입) 시 롤백 비용이 크므로** 반드시 수정 전 **전체 코드 스냅샷**을 기억해두고, 변경 전후를 명확히 분리한 브랜치 개념으로 관리.
- **경계값 결과 처리**: 1회 측정값이 목표 ±0.003 ms 이내이면 1~2회 추가 측정하여 확인. 추가 측정에서도 일관되게 목표 이하이면 달성으로 간주.
- **Correctness 실패 유형별 대응**:
  - `INCORRECT_NUMERICAL` + 오차 ~1e-2: bf16 누산 정밀도 문제 → fp32 accumulator로 복구
  - `INCORRECT_NUMERICAL` + 오차 > 1e-1: 알고리즘 오류 → 전체 롤백 후 재설계
  - `INCORRECT_SHAPE`: output 레이아웃 오류 → 즉시 롤백

---

## 12. 추가 지침

### Phase 1 (0.178 → 0.150 ms) 돌파 전략
- 현재 recurrent 구조에서의 저비용 개선 위주:
  - E1 (bf16 packed load), E2 (`ld.global.nc`), E3 (L2 persistence)
  - G1 (gate/beta fusion — 별도 kernel 제거)
  - F2 (state register prefetch 확장), F3 (gate/beta batch 로드)
  - H1 (block size 실험), H2 (launch_bounds 튜닝)
  - K1 (`sm_100a`), K2 (register 분석), K3 (`__builtin_assume`)
- **이 단계에서 chunkwise로 넘어가기 전 최대한 baseline을 낮춰두면**, chunkwise 실패 시 롤백 기준선이 더 좋아짐.

### Phase 2 (0.150 → 0.125 ms) 돌파 전략 — **가장 중요**
- **A1 → A3 → B1 → A2 → B3 → B5 → A5 → A4** 순으로 접근.
- 먼저 **작동하는 chunkwise (Tensor Core 없이, SIMT FMA로)** 를 만드는 것이 목표. C=64 고정으로 시작.
- FLA Triton reference의 수학 구조를 **직접 CUDA로 재구현** (Triton JIT 바인딩/import 금지, 로직만 이식):
  ```python
  # fla/ops/gated_delta_rule/chunk.py 의 chunk_gated_delta_rule_fwd 참조
  # fwd_prepare_wy_repr_kernel_chunk64 → WY prepare CUDA 커널로 이식
  # chunk_fwd_kernel_h → state 전이 CUDA 커널로 이식
  # chunk_fwd_kernel_o → output CUDA 커널로 이식
  ```
- Prefill 1회 동작에 **단일 커널**에 fuse할지 **분리 3-pass**로 갈지 초기 결정. 단일 fused가 SMEM bandwidth 이득 크지만 구현 복잡도 높음. 
  - **권장**: 먼저 3-pass로 작동시켜 correctness 잡고, Phase 3에서 fuse.
- 이 단계에서 correctness 검증이 가장 중요. 각 chunk 경계에서 state를 pytorch reference와 비교하는 디버그 모드 도입 권장.
- **M1 (γ_intra 인수분해), M2 (GEMM 결합 재배치), M3 (γ_cum LUT), M4 (log-space 유지), M5 (β K 선곱)은 chunkwise 재작성과 동시에 반드시 적용**한다. 이들 없이 구현하면 불필요한 C² 행렬과 중복 FLOP이 SMEM/register 예산을 소진하여 Phase 3(Tensor Core) 도입 여지가 사라진다. FLA Triton reference가 이미 M1/M2/M5를 사용하므로, 코드 이식 시 자연스럽게 따라온다 — **놓치지 않기만 하면 됨**. M3/M4는 GDN 특유의 gate 누적으로 인해 별도로 주의 깊게 설계해야 함.

### Phase 3 (0.125 → 0.110 ms) 돌파 전략
- **C1 → D1 → D2 → C2 → D3 → C5 → E6 → F1** 순.
- Tensor Core 우선: wgmma는 Hopper에서 검증된 API이므로 sm_100a에서도 안정적. tcgen05(C3)는 Phase 4로 미룸.
- TMA는 host-side TensorMap 생성 코드를 bench wrapper에 넣어야 할 수 있음 — `config.toml` 또는 python wrapper 측 변경 필요한지 먼저 확인.

### Phase 4 (0.110 → 0.100 ms) 돌파 전략
- **J1 → I1 → C3 → F4 → H5** 순.
- 이 단계는 단일 최적화로 0.010 ms 줄이기 매우 어려움. 2-3개 조합 필요.
- 2-CTA cluster(J1)는 Q 읽기 감소 + distributed SMEM broadcast 조합으로 가장 유망.
- warp specialization(I1)은 구현 복잡하지만 compute/memory overlap 이론 최대치.
- 10회 이상 반복해도 진전 없으면 NCU profiling (`ncu --set full`) 결과로 병목 재확인:
  - `smsp__inst_executed_pipe_tensor` (Tensor Core 활용도)
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` (global load)
  - `smsp__warp_issue_stalled_*` (stall reason)
- 수학적 트릭 중 남은 것: M6 (WY 대칭성, SIMT 잔존 경로 있을 때), M7 (minimax 근사, preprocess fusion 여력 있을 때), M9 (rank-C keep, 실험적).

### 작업 공정 규칙
- 최적화의 효과가 미미할 때(0.002ms 미만 개선), 여러 소규모 최적화를 조합하는 것도 고려한다.
- 하나의 최적화가 성공하면, 그 위에 다음 최적화를 쌓아 올린다 (누적).
- `modal run`의 출력을 끝까지 확인한다. 컴파일 에러가 발생하면 커널 코드를 수정하여 해결한다.
- `nvcc --resource-usage` 또는 `-Xptxas -v`를 항상 커널 수정 후 확인하여 register/spill 변화 추적.
- 대규모 변경 전 간단한 **pseudo-code 설계 노트**를 남긴다 — correctness 디버깅에 필수.
- **chunkwise 변경 후에는 반드시 가장 짧은 seq (T ≤ C) 워크로드와 가장 긴 seq 워크로드를 동시에 검증**한다. chunk 경계 처리 버그가 잦음.

---

## 13. Prefill 특화 주의사항

1. **가변 seq_len (cu_seqlens)**: batch 내 서로 다른 seq_len 처리. Padding 없이 packed layout. Grid/chunk 할당 시 반드시 고려.
2. **State out은 batch 단위**: chunk 마지막이 batch 경계와 일치해야 함 → workload partition 신중.
3. **수치 누적 오차**: T가 클수록 fp32 누산 오차 누적. bf16 accumulator 시도 시 특히 주의.
4. **Q/K/V 메모리 접근 패턴이 decode와 다름**: decode는 state가 전체 bandwidth의 대부분 — prefill은 K/V streaming이 지배적. E4 (streaming hint) 우선순위가 decode보다 훨씬 높음.
5. **Output volume 큼**: T × 8 × 128 × 2B가 batch당 수 MB. Output store path 최적화(E5, D6)가 decode 대비 더 중요.
6. **작은 T workload 주의**: T < C인 workload에서는 chunkwise가 오히려 오버헤드 → fallback 경로 또는 dynamic dispatch 필요 (A3 참조).
7. **Gate 누적 수치 범위**: 긴 T(≥4096) + 낮은 gate(<0.5) 또는 높은 gate(>0.99) 조합에서 `γ_cum`이 fp32 범위를 이탈 → M4 log-space 표현이 **correctness 문제**로 이어질 수 있음을 인지.