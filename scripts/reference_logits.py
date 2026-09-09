import json

import torch

from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)


MODEL_DIR = (
    "models/source/"
    "tinystories-llama-15m"
)

WEIGHTS_PATH = (
    f"{MODEL_DIR}/model.safetensors"
)

OUTPUT_PATH = (
    "reference_logits.json"
)


# --------------------------------------------------
# 1. Config만 읽는다.
#
# from_pretrained(model) 은 사용하지 않는다.
# --------------------------------------------------

config = AutoConfig.from_pretrained(
    MODEL_DIR,
)


# --------------------------------------------------
# 2. 모델 구조를 CPU 위에 직접 생성
#
# 이렇게 만들면 parameter들이 meta가 아니라
# 실제 CPU Tensor로 생성된다.
# --------------------------------------------------

model = AutoModelForCausalLM.from_config(
    config,
)


# --------------------------------------------------
# 3. safetensors 직접 읽기
# --------------------------------------------------

state_dict = load_file(
    WEIGHTS_PATH,
    device="cpu",
)

print(
    "checkpoint tensor count:",
    len(state_dict),
)


# --------------------------------------------------
# 4. 이 checkpoint의 tied embedding 보정
#
# 우리가 Rust importer에서 했던 것과 같은 처리.
# --------------------------------------------------

if (
    "model.embed_tokens.weight"
    not in state_dict
    and "lm_head.weight"
    in state_dict
):
    print(
        "using lm_head.weight "
        "as model.embed_tokens.weight"
    )

    state_dict[
        "model.embed_tokens.weight"
    ] = state_dict[
        "lm_head.weight"
    ]


if (
    "lm_head.weight"
    not in state_dict
    and "model.embed_tokens.weight"
    in state_dict
):
    print(
        "using model.embed_tokens.weight "
        "as lm_head.weight"
    )

    state_dict[
        "lm_head.weight"
    ] = state_dict[
        "model.embed_tokens.weight"
    ]


# --------------------------------------------------
# 5. 직접 weight 로드
# --------------------------------------------------

result = model.load_state_dict(
    state_dict,
    strict=False,
)

print(
    "missing keys:",
    result.missing_keys,
)

print(
    "unexpected keys:",
    result.unexpected_keys,
)


# tied embedding 관계도 다시 설정
model.tie_weights()

model.eval()


# --------------------------------------------------
# 6. 혹시 meta tensor가 남았는지 검증
# --------------------------------------------------

meta_parameters = [
    name
    for name, parameter
    in model.named_parameters()
    if parameter.device.type == "meta"
]

print(
    "meta parameters:",
    meta_parameters,
)

if meta_parameters:
    raise RuntimeError(
        "meta parameters still exist: "
        f"{meta_parameters}"
    )


# --------------------------------------------------
# 7. 우리와 동일한 입력
# --------------------------------------------------

input_ids = torch.tensor(
    [
        [1, 42]
    ],
    dtype=torch.long,
)


# --------------------------------------------------
# 8. Reference Forward
# --------------------------------------------------

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        use_cache=False,
    )


logits = outputs.logits

last_logits = (
    logits[
        0,
        -1,
    ]
    .float()
    .cpu()
)


# --------------------------------------------------
# 9. 결과 확인
# --------------------------------------------------

print(
    "shape:",
    tuple(
        logits.shape
    ),
)

print(
    "range:",
    float(
        last_logits.min()
    ),
    float(
        last_logits.max()
    ),
)

argmax_token = int(
    torch.argmax(
        last_logits
    )
)

print(
    "argmax token:",
    argmax_token,
)


top_values, top_indices = (
    torch.topk(
        last_logits,
        k=5,
    )
)

print(
    "top 5:"
)

for (
    token_id,
    value,
) in zip(
    top_indices.tolist(),
    top_values.tolist(),
):
    print(
        f"token {token_id:5d}"
        f" => {value}"
    )


# --------------------------------------------------
# 10. 전체 32,000 logits 저장
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        last_logits.tolist(),
        file,
    )


print(
    f"saved {OUTPUT_PATH}"
)
