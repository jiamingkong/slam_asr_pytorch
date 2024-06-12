from datasets import load_dataset
import soundfile as sf
import torch
from modeling.asr import SLAM_ASR
from safetensors.torch import load_file


asr = SLAM_ASR(
    "facebook/hubert-base-ls960",
    "TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    train_mode="adapter",
)
# load the state_dict from output/adapter_weights.pt
adapter_weight = load_file("output/checkpoint-1750/model.safetensors")
asr.load_state_dict(adapter_weight, strict=False)


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
    trust_remote_code=True,
)
ds = ds.map(map_to_array)

for i in range(len(ds)):
    x = ds["speech"][i]
    y = ds["text"][i]
    # asr(x)
    output = asr.generate(x)  # causal of shape (b, seq_len, vocab_size)
    print(f"Predicted: {asr.language_tokenizer.batch_decode(output)[0]}")
    print(f"Reference: {y.lower()}")
    print("\n\n")
