# RustLLM

Apple Metal GPU에서 소형 Transformer 언어 모델의 **추론 과정**을 직접 구현해 보는 Rust 프로젝트입니다. 텐서 연산, Metal 커널, Transformer 블록, 모델 가중치 로딩, 문자 단위 토크나이저, greedy 토큰 생성을 한 저장소에서 다룹니다. 현재 기본 실행은 Hugging Face Llama safetensors 가중치를 이 프로젝트의 `model.bin` 형식으로 변환합니다.

`models/tiny`는 내부 추론 경로 확인을 위한 매우 작은 예제 모델입니다. 일반적인 대화형 언어 모델의 품질을 목표로 한 모델은 아닙니다. 기본 `cargo run`은 이 예제 모델을 실행하지 않고 TinyStories Llama 가중치를 변환합니다.

## 현재 구현된 기능

- Metal 기반 `f32` 텐서 저장소와 shape/stride 관리
- Metal 셰이더 기반 연산
  - 덧셈, 원소별 곱셈, SiLU, RMSNorm, Softmax
  - 행렬 곱셈(naive 및 tile 8/16/32 구현)
  - batched matrix multiplication, embedding lookup, contiguous materialization
  - attention scale/mask 및 RoPE(Rotary Position Embedding)
- Decoder-only Transformer 추론
  - Token embedding, multi-head self-attention, RMSNorm
  - SwiGLU MLP, residual connection, LM head
- `config.json`과 커스텀 바이너리 `model.bin` 가중치 포맷 로딩 및 shape 검증
- JSON 기반 문자(character) 단위 토크나이저
- argmax를 사용하는 greedy autoregressive generation 및 EOS 종료 처리

## 동작 환경

- macOS 및 Apple Metal 지원 GPU가 필요합니다.
- Rust 2024 edition을 사용합니다. 설치되지 않았다면 [rustup](https://rustup.rs/)으로 Rust 툴체인을 설치하세요.
- Metal 기반 **추론**은 macOS와 Apple Metal 지원 GPU가 필요합니다. 가중치 변환만 수행하는 기본 `cargo run`은 Metal GPU 없이도 실행할 수 있습니다.

## 빠른 시작

저장소 루트에서 실행합니다.

```bash
cargo run
```

프로그램은 다음 순서로 동작합니다.

1. `models/source/tinystories-llama-15m/config.json`을 읽고 내부 `ModelConfig`로 변환합니다.
2. `model.safetensors`의 F32/F16/BF16 텐서를 읽습니다.
3. Llama 가중치 이름과 선형 계층의 행렬 방향을 내부 형식으로 변환합니다.
4. `models/tinystories-llama-15m/config.json` 및 `model.bin`을 저장합니다.
5. 저장한 `model.bin`을 다시 읽어 텐서 개수를 검증합니다.

다른 경로의 모델을 변환하려면 입력 모델 디렉터리와 출력 디렉터리를 순서대로 전달합니다.

```bash
cargo run -- path/to/source-model path/to/output-model
```

입력 디렉터리에는 `config.json`과 `model.safetensors`가 필요합니다. 현재 변환기는 bias 없는 multi-head attention만 지원하며, `num_key_value_heads`가 `num_attention_heads`와 다른 GQA/MQA 모델 또는 bias 텐서가 있는 모델은 명확한 오류로 중단합니다.

## 예제 모델 구성

포함된 tiny 모델의 설정은 다음과 같습니다.

| 항목 | 값 |
| --- | ---: |
| Vocabulary size | 4 |
| Hidden size | 4 |
| Intermediate size | 8 |
| Transformer layers | 1 |
| Attention heads | 2 |
| Maximum sequence length | 128 |
| RMSNorm epsilon | 0.00001 |
| RoPE theta | 10000.0 |

## 프로젝트 구조

```text
.
├── kernels/             # Metal compute shader 소스
├── models/source/       # Hugging Face 형식 Llama 입력 모델
├── models/tinystories-llama-15m/
│   ├── config.json      # 변환된 내부 모델 설정
│   └── model.bin        # 변환된 가중치
├── src/
│   ├── generation/      # greedy sampler와 생성 루프
│   ├── metal/           # Metal device, command queue, pipeline cache
│   ├── model/           # 설정, 가중치 포맷, Transformer 조립
│   ├── nn/              # Embedding, Attention, MLP, RMSNorm 등 계층
│   ├── ops/             # 텐서 연산과 Metal kernel 호출
│   ├── tensor/          # Tensor, shape, stride, storage
│   ├── tokenizer/       # 문자 단위 JSON 토크나이저
│   ├── error.rs         # 프로젝트 오류 타입
│   └── main.rs          # 예제 추론 실행 진입점
├── Cargo.toml
└── README.md
```

## 모델 파일 형식

모델 디렉터리에는 아래 파일이 필요합니다.

```text
models/<model-name>/
├── config.json
├── model.bin
└── tokenizer.json
```

- `config.json`: vocabulary, hidden size, layer/head 수, 최대 시퀀스 길이, RMSNorm/RoPE 설정을 담습니다.
- `model.bin`: 프로젝트의 `TMLLWGHT` magic과 version 1을 사용하는 커스텀 `f32` 가중치 포맷입니다. 텐서 이름, shape, 값의 수를 검증하며 필요한 가중치가 없거나 shape가 다르면 로딩을 중단합니다.
- `tokenizer.json`: 현재 `type: "char"` 형식만 지원합니다. `vocab`, 선택적 `bos_token_id`, `eos_token_id`, `unk_token_id`를 포함합니다.

기본 실행은 Llama safetensors를 직접 변환합니다. `tie_word_embeddings`로 인해 `model.embed_tokens.weight`가 생략되고 `lm_head.weight`만 있는 체크포인트도 지원합니다. 변환된 모델의 추론에는 현재 프로젝트 형식의 `tokenizer.json`이 추가로 필요합니다. 원본 Llama의 `tokenizer.model`(SentencePiece)은 아직 지원하지 않으므로, 변환 완료만으로 텍스트 생성까지 가능한 것은 아닙니다.

## 테스트 및 확인

```bash
cargo test
cargo run
```

`cargo test`는 가중치 공유(`lm_head.weight`) 체크포인트의 변환과 미지원 GQA 모델의 거부를 확인합니다. `cargo run`은 변환 후 원본·변환 텐서 개수 및 저장된 결과의 텐서 개수를 출력합니다. 가중치 변환 자체에는 Metal GPU가 필요하지 않습니다.

실행 시에는 `source tensors`, `converted tensors`, `verified`가 차례로 출력되는지 확인합니다. 문제가 생기면 아래를 우선 확인하세요.

- 모델 파일 오류: 입력 디렉터리 아래 `config.json`, `model.safetensors`가 모두 있는지 확인
- 모델 구조 오류: GQA/MQA 모델인지 `num_key_value_heads`와 `num_attention_heads`를 확인
- dtype 오류: safetensors의 텐서가 F32, F16 또는 BF16인지 확인
- 추론 준비 오류: 변환 결과에 맞는 프로젝트 형식 `tokenizer.json`이 별도로 준비됐는지 확인

## 현재 한계와 다음 개선 방향

- 학습(training), fine-tuning, 모델 다운로드 기능은 포함하지 않습니다.
- 문자 단위 토크나이저만 지원하며 BPE/SentencePiece는 지원하지 않습니다.
- 생성은 greedy decoding만 지원하며 temperature, top-k/top-p sampling은 없습니다.
- KV cache가 없어 생성 단계마다 전체 토큰 시퀀스를 다시 forward 합니다.
- `f16` 추론, 배치 추론, CLI 인자 처리, 자동화된 단위/통합 테스트는 다음 단계의 개선 항목입니다.

## 기술 스택

- Rust 2024
- [`metal`](https://crates.io/crates/metal) crate
- Metal Shading Language
- `serde`, `serde_json`
