# RustLLM

Apple Metal GPU에서 소형 Transformer 언어 모델의 **추론 과정**을 직접 구현해 보는 Rust 프로젝트입니다. 텐서 연산, Metal 커널, Transformer 블록, Llama 가중치 로딩, SentencePiece/Hugging Face 토크나이저, autoregressive 토큰 생성을 한 저장소에서 다룹니다.

`models/tiny`는 내부 추론 경로 확인을 위한 매우 작은 예제 모델입니다. Hugging Face에서 받거나 변환한 실제 모델은 Git에 포함하지 않으며, 각 개발자가 아래 방법으로 내려받습니다.

## 현재 구현된 기능

- Metal 기반 F32/F16 텐서 저장소와 shape/stride 관리, 모델 전체 dtype 변환
- Metal 셰이더 기반 연산
  - 덧셈, 원소별 곱셈, SiLU, RMSNorm, Softmax
  - F32 행렬 곱셈과 F16 입력/F32 누산/F16 출력 MatMul MVP
  - batched matrix multiplication, embedding lookup, contiguous materialization
  - attention scale/mask 및 RoPE(Rotary Position Embedding)
- Decoder-only Transformer 추론
  - Token embedding, multi-head self-attention, RMSNorm
  - SwiGLU MLP, residual connection, LM head
- `config.json`과 커스텀 바이너리 `model.bin` 가중치 포맷 로딩 및 shape 검증
- `tokenizer.model` 기반 SentencePiece와 `tokenizer.json` 기반 Hugging Face encode/decode
- greedy 및 temperature/top-k/top-p seeded autoregressive generation, EOS 종료 처리
- 레이어별 Key/Value cache를 이용한 토큰 단위 디코딩
- import 시 지원되는 원본 토크나이저 파일을 변환 모델 폴더에 함께 복사
- 모델 설정을 빠르게 확인하는 `inspect` 명령

## 동작 환경

- macOS 및 Apple Metal 지원 GPU가 필요합니다.
- Rust 2024 edition을 사용합니다. 설치되지 않았다면 [rustup](https://rustup.rs/)으로 Rust 툴체인을 설치하세요.
- Metal 기반 **추론**은 macOS와 Apple Metal 지원 GPU가 필요합니다. `cargo run -- import ...` 가중치 변환은 Metal GPU 없이도 실행할 수 있습니다.

## 실제 TinyStories 이야기 생성

저장소 루트에서 실행합니다.

먼저 Hugging Face CLI를 설치하고 모델을 내려받습니다. 예시는 현재 확인한 110M 모델 기준입니다.

```bash
uv tool install "huggingface_hub[cli]"
hf download Xenova/llama2.c-stories110M --local-dir models/source/llama2.c-stories110M
```

그 다음 프로젝트 내부 포맷으로 변환합니다.

```bash
cargo run -- import \\
  --source models/source/llama2.c-stories110M \\
  --output models/llama2.c-stories110M
```

변환 후 생성 명령을 실행합니다.

```bash
cargo run -- run --model models/llama2.c-stories110M --prompt "Once upon a time" --max-tokens 64
```

이 명령은 아래 전체 경로를 실행합니다.

```text
"Once upon a time"
        ↓
SentencePiece encode (BOS 포함)
        ↓
TinyStories 110M 가중치 + Rust Transformer + Metal kernels
        ↓
생성된 token IDs (EOS면 종료)
        ↓
SentencePiece decode
        ↓
영어 이야기 출력
```

`run`은 tokenizer와 모델의 vocabulary 크기를 대조합니다. 프롬프트는 한 번만 전체 forward하고, 이후 생성 토큰은 레이어별 KV cache에 Key/Value를 누적해 한 토큰씩 forward합니다. 따라서 이전 토큰 전체를 매번 다시 계산하지 않습니다. 프롬프트와 생성 토큰의 총길이는 모델 context length를 넘지 않도록 제한됩니다. 상태 메시지는 stderr로, 생성된 이야기만 stdout으로 출력합니다.

## Hugging Face 모델 변환

Hugging Face 모델을 `hf download <repo-id> --local-dir models/source/<model-name>`으로 내려받은 뒤 변환합니다. 모델 파일은 의도적으로 `.gitignore`에 포함되어 있으므로 Git에 추가하지 않습니다.

변환기는 Llama config/safetensors를 내부 포맷으로 바꾸고 `config.json`, `model.bin` 및 원본에 있는 토크나이저 관련 파일을 출력 폴더에 준비합니다. 지원 파일은 `tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`입니다.

다른 경로의 모델을 변환하려면 입력 모델 디렉터리와 출력 디렉터리를 순서대로 전달합니다.

```bash
cargo run -- import --source path/to/source-model --output path/to/output-model
```

`--weight-format`은 파일 안의 가중치 저장 형식을 고릅니다. F16 모델은 실행 중 F32 전체 모델을 GPU에서 F16으로 캐스팅하지 않고 F16 텐서로 직접 로드합니다. INT8은 Linear weight만 INT8과 채널별 F32 scale로 저장하며, embedding/RMSNorm과 runtime activation은 F16을 유지합니다.

```bash
cargo run -- import \
  --source path/to/source-model \
  --output path/to/output-model-f16 \
  --weight-format f16
```

INT8 weight-only 패키지는 다음처럼 변환합니다.

```bash
cargo run -- import \
  --source models/source/SmolLM2-135M \
  --output models/SmolLM2-135M-int8 \
  --weight-format int8
```

입력 디렉터리에는 `config.json`과 `model.safetensors`가 필요합니다. 현재 변환기는 bias 없는 Llama 계열 multi-head attention과 GQA/MQA를 지원하며, bias 텐서가 있는 모델은 명확한 오류로 중단합니다.

## 명령어와 생성 옵션

모델의 설정만 확인할 때는 Metal GPU 없이 다음 명령을 실행할 수 있습니다.

```bash
cargo run -- inspect --model models/llama2.c-stories110M
```

`run`은 기본적으로 greedy decoding을 사용합니다. `--temperature`가 `0`보다 클 때만 확률 샘플링이 활성화되며, 재현 가능한 결과를 위해 `--seed`를 지정할 수 있습니다.

```bash
cargo run -- run \\
  --model models/llama2.c-stories110M \\
  --prompt "Once upon a time" \\
  --dtype f16 \\
  --max-tokens 64 \\
  --temperature 0.8 \\
  --top-k 40 \\
  --top-p 0.95 \\
  --seed 42 \\
  --metrics
```

- `--dtype f32|f16`: runtime activation dtype입니다. 기본값은 `f32`이며 F16 dense 모델과 INT8 weight-only 모델은 모두 `--dtype f16`으로 실행합니다. INT8은 activation dtype이 아닙니다.
- `--max-tokens`: 생성할 최대 새 토큰 수입니다. 모델의 남은 context length를 넘지 않습니다.
- `--temperature`: `0`이면 greedy decoding, 양수이면 확률 샘플링을 사용합니다.
- `--top-k`, `--top-p`: 샘플링 후보를 각각 상위 K개 및 누적 확률 P로 제한합니다.
- `--metrics`: prefill, decode, 전체 생성 시간과 속도를 stderr에 출력합니다.
- `--benchmark`: 스트리밍 출력과 flush를 끄고 model size, load time, prefill/decode 속도, 전체 generation time을 출력합니다. F16과 INT8 비교에 사용합니다.

릴리스 빌드 뒤 같은 prompt와 token 수로 각각 여러 번 실행합니다. 첫 실행은 Metal pipeline 및 파일 cache warmup 용도로 버리고, 이후 3회의 수치를 평균내면 됩니다.

```bash
cargo build --release

./target/release/tiny-metal-llm run \
  --model models/SmolLM2-135M \
  --dtype f16 \
  --prompt "Once upon a time" \
  --max-tokens 128 \
  --benchmark

./target/release/tiny-metal-llm run \
  --model models/SmolLM2-135M-int8 \
  --dtype f16 \
  --prompt "Once upon a time" \
  --max-tokens 128 \
  --benchmark
```

프로세스 최대 RSS는 Rust 코드가 아니라 macOS 도구로 별도로 확인합니다.

```bash
/usr/bin/time -l ./target/release/tiny-metal-llm run \
  --model models/SmolLM2-135M-int8 \
  --dtype f16 \
  --prompt "Once upon a time" \
  --max-tokens 128 \
  --benchmark
```

## 예제 모델 구성

포함된 tiny 모델의 설정은 다음과 같습니다.

| 항목                    |      값 |
| ----------------------- | ------: |
| Vocabulary size         |       4 |
| Hidden size             |       4 |
| Intermediate size       |       8 |
| Transformer layers      |       1 |
| Attention heads         |       2 |
| Maximum sequence length |     128 |
| RMSNorm epsilon         | 0.00001 |
| RoPE theta              | 10000.0 |

## 프로젝트 구조

```text
.
├── kernels/             # Metal compute shader 소스
├── models/              # 로컬 모델 저장소 (Git 제외)
│   ├── source/          # Hugging Face 형식 Llama 입력 모델
│   └── <model-name>/    # 변환된 내부 모델
├── models/tiny/         # Git에 포함되는 1.5 KB 테스트 픽스처
│   ├── config.json      # 변환된 내부 모델 설정
│   ├── model.bin        # 변환된 가중치
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
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

## 최근 리팩터링

기능 동작과 모델 파일 포맷을 유지한 채, 책임이 섞여 있던 코드를 다음과 같이 분리했습니다.

- Transformer의 KV Cache 추론 경로를 `src/model/transformer/cache.rs`로 분리해 모델 조립·가중치 로딩과 실행 경로를 구분했습니다.
- `ModelWeights`는 컬렉션 검증에 집중하고, `src/model/weight_codec.rs`가 `model.bin`의 저장·로드를 담당하도록 분리했습니다.
- `Tensor`의 원소별 GPU 연산(`add`, `mul`, `SiLU`)을 `src/tensor/elementwise.rs`로 분리하고 dtype·shape·contiguous 처리를 공통화했습니다.

## 모델 파일 형식

모델 디렉터리에는 아래 파일이 필요합니다.

```text
models/<model-name>/
├── config.json
├── model.bin
├── tokenizer.model          # 선택: SentencePiece
├── tokenizer.json           # 선택: Hugging Face Tokenizers
└── tokenizer_config.json    # 선택: BOS/EOS 메타데이터
```

- `config.json`: vocabulary, hidden size, layer/head 수, 최대 시퀀스 길이, RMSNorm/RoPE 설정을 담습니다.
- `model.bin`: 프로젝트의 `TMLLWGHT` magic과 version 1을 사용하는 커스텀 `f32` 가중치 포맷입니다. 텐서 이름, shape, 값의 수를 검증하며 필요한 가중치가 없거나 shape가 다르면 로딩을 중단합니다.
- `tokenizer.model`: 원본 Llama SentencePiece 모델입니다. 있으면 우선 사용합니다.
- `tokenizer.json`: Hugging Face Tokenizers 형식입니다. `tokenizer.model`이 없을 때 사용합니다.
- `tokenizer_config.json`: 토크나이저별 BOS/EOS 메타데이터를 읽습니다.

`run`은 선택된 토크나이저로 프롬프트를 ID로 바꾸고, 생성 ID를 문자열로 되돌립니다. `tokenizer.model`과 `tokenizer.json` 중 하나는 반드시 필요합니다.

`tie_word_embeddings`로 인해 `model.embed_tokens.weight`가 생략되고 `lm_head.weight`만 있는 체크포인트도 지원합니다. 변환 결과에 지원되는 토크나이저 파일까지 복사되므로 변환 직후 `run`으로 실제 텍스트 생성을 실행할 수 있습니다.

## 테스트 및 확인

```bash
cargo test
cargo run -- run --model models/llama2.c-stories110M --prompt "Once upon a time"
```

`cargo test`는 텐서/Metal 연산, Transformer, KV Cache, 샘플러, 토크나이저 선택과 가중치 파일 저장·로드를 확인합니다. Hugging Face 모델이 필요한 SentencePiece 통합 테스트는 모델을 내려받은 뒤 `cargo test -- --ignored`로 실행합니다. `run`은 Apple Metal GPU가 필요합니다.

`import` 실행 시에는 `source tensors`, `converted tensors`, `copied <tokenizer-file>`이 출력되는지 확인합니다. 문제가 생기면 아래를 우선 확인하세요.

- 모델 파일 오류: 입력 디렉터리 아래 `config.json`, `model.safetensors`가 모두 있는지 확인
- 모델 구조 오류: GQA/MQA 모델인지 `num_key_value_heads`와 `num_attention_heads`를 확인
- dtype 오류: safetensors의 텐서가 F32, F16 또는 BF16인지 확인
- 토크나이저 오류: 변환 결과에 `tokenizer.model`과 `tokenizer_config.json`이 있는지 확인하고, 없다면 `import`를 다시 실행

## 현재 한계와 다음 개선 방향

- 학습(training), fine-tuning, 모델 다운로드 기능은 포함하지 않습니다.
- KV cache는 레이어별 고정 크기 버퍼에 K/V를 기록합니다.
- 샘플링은 CPU에서 수행하므로, 대규모 vocabulary 모델에서는 병목이 될 수 있습니다.
- 배치 추론과 자동화된 GPU 통합 테스트는 다음 단계의 개선 항목입니다.

## 기술 스택

- Rust 2024
- [`metal`](https://crates.io/crates/metal) crate
- Metal Shading Language
- `serde`, `serde_json`
