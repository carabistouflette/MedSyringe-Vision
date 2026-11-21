#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {
  local name="$1" repository="$2" revision="$3"
  shift 3
  local folder="data/models/$name"
  mkdir -p "$folder"
  for file in "$@"; do
    if [ ! -f "$folder/$file" ]; then
      curl --fail --location --silent --show-error --retry 2 --max-time 600 \
        "https://huggingface.co/$repository/resolve/$revision/$file?download=true" \
        -o "$folder/$file.part"
      mv "$folder/$file.part" "$folder/$file"
    fi
  done
}
text_models=(config.json preprocessor_config.json tokenizer_config.json special_tokens_map.json added_tokens.json)
fetch dino IDEA-Research/grounding-dino-tiny a2bb814dd30d776dcf7e30523b00659f4f141c71 "${text_models[@]}" tokenizer.json vocab.txt model.safetensors
fetch owl google/owlv2-base-patch16 2a1560802f8cf3c408fec9b809d705f56a2f7146 "${text_models[@]}" merges.txt vocab.json model.safetensors
fetch dinov2-small facebook/dinov2-small ed25f3a31f01632728cabb09d1542f84ab7b0056 config.json model.safetensors

verifier="data/models/verifier/mobilenet_v3_small-047dcff4.pth"
if [ ! -f "$verifier" ]; then
  mkdir -p "$(dirname "$verifier")"
  curl --fail --location --silent --show-error --retry 2 --max-time 600 \
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o "$verifier.part"
  mv "$verifier.part" "$verifier"
fi
