# RustLLM

Apple Metal GPU에서 소형 Transformer 언어 모델의 **추론 과정**을 직접 구현해 보는 Rust 프로젝트입니다. 텐서 연산, Metal 커널, Transformer 블록, 모델 가중치 로딩, 문자 단위 토크나이저, greedy 토큰 생성을 한 저장소에서 다룹니다.

현재 포함된 `models/tiny`는 동작 확인을 위한 매우 작은 예제 모델입니다. 일반적인 대화형 언어 모델의 품질을 목표로 한 모델은 아닙니다.

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
- Linux, Windows 또는 Metal GPU가 없는 환경에서는 현재 실행할 수 없습니다. `Metal GPU를 찾을 수 없습니다.` 오류는 이 환경 제약에 해당합니다.

## 빠른 시작

저장소 루트에서 실행합니다.

```bash
cargo run
```

프로그램은 다음 순서로 동작합니다.

1. `models/tiny/config.json`과 `models/tiny/model.bin`에서 모델을 로드합니다.
2. `models/tiny/tokenizer.json`에서 문자 단위 토크나이저를 로드합니다.
3. `src/main.rs`의 기본 프롬프트 `"ab"`를 토큰화합니다.
4. 최대 5개의 토큰을 greedy 방식으로 생성합니다.
5. 전체 토큰, 복원 텍스트, 생성된 이어쓰기 텍스트를 출력합니다.

`src/main.rs`의 `prompt`와 `generate_greedy`의 최대 생성 토큰 수를 변경하면 다른 입력을 시험할 수 있습니다. 입력 문자는 현재 토크나이저 vocabulary에 있어야 하며, 없을 경우 `unk_token_id`가 설정되어 있지 않으면 오류가 발생합니다.

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
├── models/tiny/         # 예제 모델 설정, 가중치, 토크나이저
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

새 모델을 연결하려면 현재 Transformer가 요구하는 텐서 이름과 shape에 맞춰 가중치를 변환해야 합니다. Hugging Face 형식의 모델을 직접 불러오는 기능은 아직 없습니다.

## 테스트 및 확인

```bash
cargo test
cargo run
```

2026-09-09 기준으로 `cargo test`는 성공했습니다. 현재 자동화된 unit test는 아직 없어서 실행 결과는 `0 passed; 0 failed`입니다. `cargo run`은 Metal 지원 macOS 장비에서 실행해 확인해야 합니다.

실행 시에는 프롬프트 토큰, 전체 토큰 배열, 디코딩 텍스트, continuation이 출력되는지 확인합니다. 문제가 생기면 아래를 우선 확인하세요.

- Metal GPU 오류: macOS와 Metal 지원 GPU에서 실행 중인지 확인
- 모델 파일 오류: `models/tiny` 아래 세 파일이 모두 존재하는지 확인
- vocabulary 오류: `config.json`의 `vocab_size`와 `tokenizer.json`의 vocab 길이가 일치하는지 확인
- 입력 오류: 프롬프트가 비어 있지 않고 최대 시퀀스 길이를 넘지 않는지 확인

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
