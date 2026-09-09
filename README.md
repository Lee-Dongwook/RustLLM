# RustLLM

Apple Metal GPU에서 소형 Transformer 언어 모델의 **추론 과정**을 직접 구현해 보는 Rust 프로젝트입니다. 텐서 연산, Metal 커널, Transformer 블록, Llama 가중치 로딩, SentencePiece 토크나이저, greedy 토큰 생성을 한 저장소에서 다룹니다.

`models/tiny`는 내부 추론 경로 확인을 위한 매우 작은 예제 모델입니다. 실제 영어 이야기 생성은 `tinystories-llama-15m` 모델을 사용합니다.

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
- Llama `tokenizer.model` 기반 SentencePiece encode/decode
- argmax를 사용하는 greedy autoregressive generation 및 EOS 종료 처리
- import 시 원본 SentencePiece 파일을 변환 모델 폴더에 함께 복사

## 동작 환경

- macOS 및 Apple Metal 지원 GPU가 필요합니다.
- Rust 2024 edition을 사용합니다. 설치되지 않았다면 [rustup](https://rustup.rs/)으로 Rust 툴체인을 설치하세요.
- Metal 기반 **추론**은 macOS와 Apple Metal 지원 GPU가 필요합니다. `cargo run -- import ...` 가중치 변환은 Metal GPU 없이도 실행할 수 있습니다.

## 실제 TinyStories 이야기 생성

저장소 루트에서 실행합니다.

처음 실행하거나 `models/tinystories-llama-15m`에 `tokenizer.model`이 없다면, 아래의 `import` 명령을 먼저 한 번 실행합니다.

```bash
cargo run -- run \\
  --model models/tinystories-llama-15m \\
  --prompt "Once upon a time" \\
  --max-tokens 64
```

이 명령은 아래 전체 경로를 실행합니다.

```text
"Once upon a time"
        ↓
SentencePiece encode (BOS 포함)
        ↓
TinyStories 15M 가중치 + Rust Transformer + Metal kernels
        ↓
생성된 token IDs (EOS면 종료)
        ↓
SentencePiece decode
        ↓
영어 이야기 출력
```

`run`은 tokenizer와 모델의 vocabulary 크기를 대조합니다. 프롬프트와 생성 토큰의 총길이는 모델 context length를 넘지 않도록 제한됩니다. 상태 메시지는 stderr로, 생성된 이야기만 stdout으로 출력합니다.

## Hugging Face 모델 변환

```bash
cargo run -- import \\
  --source models/source/tinystories-llama-15m \\
  --output models/tinystories-llama-15m
```

변환기는 Llama config/safetensors를 내부 포맷으로 바꾸고 `config.json`, `model.bin`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`을 출력 폴더에 준비합니다.

다른 경로의 모델을 변환하려면 입력 모델 디렉터리와 출력 디렉터리를 순서대로 전달합니다.

```bash
cargo run -- import --source path/to/source-model --output path/to/output-model
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
│   ├── tokenizer/       # SentencePiece와 문자 토크나이저 구현
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
├── tokenizer.model
├── tokenizer_config.json
└── special_tokens_map.json
```

- `config.json`: vocabulary, hidden size, layer/head 수, 최대 시퀀스 길이, RMSNorm/RoPE 설정을 담습니다.
- `model.bin`: 프로젝트의 `TMLLWGHT` magic과 version 1을 사용하는 커스텀 `f32` 가중치 포맷입니다. 텐서 이름, shape, 값의 수를 검증하며 필요한 가중치가 없거나 shape가 다르면 로딩을 중단합니다.
- `tokenizer.model`: 원본 Llama SentencePiece 모델입니다. `run`은 이 파일로 프롬프트를 ID로 바꾸고, 생성 ID를 문자열로 되돌립니다.
- `tokenizer_config.json`: `add_bos_token`과 `add_eos_token`을 읽습니다. TinyStories는 입력 시작에 BOS를 붙입니다.
- `special_tokens_map.json`: 원본 모델의 special token 메타데이터입니다.

`tie_word_embeddings`로 인해 `model.embed_tokens.weight`가 생략되고 `lm_head.weight`만 있는 체크포인트도 지원합니다. 변환 결과에 SentencePiece 파일까지 복사되므로 변환 직후 `run`으로 실제 텍스트 생성을 실행할 수 있습니다.

## 테스트 및 확인

```bash
cargo test
cargo run -- run --model models/tinystories-llama-15m --prompt "Once upon a time"
```

`cargo test`는 가중치 공유(`lm_head.weight`) 체크포인트의 변환, 미지원 GQA 모델의 거부, 그리고 포함된 TinyStories SentencePiece 모델의 encode/decode round trip을 확인합니다. `run`은 Apple Metal GPU가 필요합니다.

`import` 실행 시에는 `source tensors`, `converted tensors`, `copied tokenizer.model`이 출력되는지 확인합니다. 문제가 생기면 아래를 우선 확인하세요.

- 모델 파일 오류: 입력 디렉터리 아래 `config.json`, `model.safetensors`가 모두 있는지 확인
- 모델 구조 오류: GQA/MQA 모델인지 `num_key_value_heads`와 `num_attention_heads`를 확인
- dtype 오류: safetensors의 텐서가 F32, F16 또는 BF16인지 확인
- 토크나이저 오류: 변환 결과에 `tokenizer.model`과 `tokenizer_config.json`이 있는지 확인하고, 없다면 `import`를 다시 실행

## 현재 한계와 다음 개선 방향

- 학습(training), fine-tuning, 모델 다운로드 기능은 포함하지 않습니다.
- 생성은 greedy decoding만 지원하며 temperature, top-k/top-p sampling은 없습니다.
- KV cache가 없어 생성 단계마다 전체 토큰 시퀀스를 다시 forward 합니다.
- `f16` 추론, 배치 추론, CLI 인자 처리, 자동화된 단위/통합 테스트는 다음 단계의 개선 항목입니다.

## 기술 스택

- Rust 2024
- [`metal`](https://crates.io/crates/metal) crate
- Metal Shading Language
- `serde`, `serde_json`
