"""
The main body of the ASR model,

User: <Speech> <Prompt>
Model: <Transcription>
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List

try:
    from .speech_encoder import SpeechEncoder
except ImportError:
    from speech_encoder import SpeechEncoder


class SLAM_ASR(nn.Module):
    def __init__(
        self,
        speech_encoder_model_id,
        language_model_id,
        downsample_K=5,
        hidden_dim=2048,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(SLAM_ASR, self).__init__()
        self.device = device
        """
                       |------|
                       |hubert|
                       |------|
                           |
                           |
                       (adapter, the only trainable part here)
                           |
                           v
        LLAMA("<user>: [______], transcribe it, <assistant>") --> "The weather is good"

        """

        self.language_tokenizer = LlamaTokenizer.from_pretrained(language_model_id)
        self.language_model = LlamaForCausalLM.from_pretrained(
            language_model_id,
            trust_remote_code=True,
        ).to(self.device)
        language_project_dim = self.language_model.config.hidden_size

        self.speech_encoder = SpeechEncoder(
            speech_encoder_model_id,
            language_project_dim,
            downsample_K=downsample_K,
            hidden_dim=hidden_dim,
            train_mode=train_mode,
            device=device,
        ).to(self.device)

        self.set_gradient(train_mode)
        # self.prompt_template = """<|im_start|>user\n{audio}, transcribe the audio.<|im_end|>\n<|im_start|>assistant\n"""
        self.prompt_part1 = """<|im_start|>user\n"""
        self.prompt_part2 = (
            """, transcribe the audio<|im_end|>\n<|im_start|>assistant\n"""
        )
        self.embed_bank = {"embed1": None, "embed2": None, "att1": None, "att2": None}
        self.set_embed_bank()

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.gradient_checkpointing_enable(**kwargs)

    def set_embed_bank(self, batch_size=1):
        input_dict1 = self.language_tokenizer(
            [self.prompt_part1], return_tensors="pt"
        ).to(self.device)
        input_dict2 = self.language_tokenizer(
            [self.prompt_part2], return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # precache the embeddings for prompt
        with torch.no_grad():
            inputs_embeds1 = self.language_model.model.embed_tokens(
                input_dict1.input_ids
            )
            inputs_embeds2 = self.language_model.model.embed_tokens(
                input_dict2.input_ids
            )
        self.embed_bank["embed1"] = inputs_embeds1
        self.embed_bank["embed2"] = inputs_embeds2
        self.embed_bank["att1"] = input_dict1.attention_mask
        self.embed_bank["att2"] = input_dict2.attention_mask
        print("[Preloaded embeddings for both part of the prompt.]")
        print(
            f"    {self.prompt_part1}        {inputs_embeds1.shape}\n    {self.prompt_part2}        {inputs_embeds2.shape}"
        )

    def set_gradient(self, train_mode):
        assert train_mode in ["adapter", "full"]

        # call set_gradient for speech encoder
        self.speech_encoder.set_gradient(train_mode)

        # freeze the whole language_model
        if train_mode != "full":
            for param in self.language_model.parameters():
                param.requires_grad = False
        # now list all parameters that require grad
        print("Parameters that require grad:")

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.shape}")

    def _prepare_input_embeds(
        self, audios: List[float], transcriptions: List[str] = None
    ):
        """
        First, run audios through speech_encoder to get the embeddings and mask
        """
        speech_output, mask = self.speech_encoder(audios)
        batch_size = speech_output.shape[0]
        # get the prompt embeddings
        embed1 = self.embed_bank["embed1"].repeat(batch_size, 1, 1)  # (b, 4, 2048)
        embed2 = self.embed_bank["embed2"].repeat(batch_size, 1, 1)  # (b, 11, 2048)
        att1 = self.embed_bank["att1"].repeat(batch_size, 1)
        att2 = self.embed_bank["att2"].repeat(batch_size, 1)

        true_labels = None

        prompt_length = embed1.shape[1] + speech_output.shape[1] + embed2.shape[1]
        if transcriptions is not None:
            _labels = self.language_tokenizer(
                transcriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
            labels_embeds = self.language_model.model.embed_tokens(_labels.input_ids)
            att3 = _labels.attention_mask
            prompt_embed = torch.cat(
                [embed1, speech_output, embed2, labels_embeds], dim=1
            )  # (b, 4+audio+11+seq_len, 2048)
            prompt_mask = torch.cat([att1, mask, att2, att3], dim=1)

            true_labels = _labels.input_ids
            # attach "prompt_length" * -100 to the start of the true_labels
            true_labels = torch.cat(
                [
                    torch.full(
                        (batch_size, prompt_length),
                        -100,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    true_labels,
                ],
                dim=1,
            )
        else:
            prompt_embed = torch.cat(
                [embed1, speech_output, embed2], dim=1
            )  # (b, 4+audio+11, 2048)
            prompt_mask = torch.cat([att1, mask, att2], dim=1)
            true_labels = None
        return prompt_embed, prompt_mask, true_labels

    def forward(self, audios: List[float], transcriptions: List[str] = None):
        prompt_embed, prompt_mask, true_labels = self._prepare_input_embeds(
            audios, transcriptions
        )
        # run the prompt through the language model

        outputs = self.language_model(
            inputs_embeds=prompt_embed,
            attention_mask=prompt_mask.bool(),
            labels=true_labels,
        )  # CausalLMOutputWithPast
        return outputs

    def generate(self, audios: List[str], stopping_criteria=None):
        """
        Generate the transcription
        """
        prompt_embed, prompt_mask, _ = self._prepare_input_embeds(audios)
        outputs = self.language_model.generate(
            inputs_embeds=prompt_embed,
            attention_mask=prompt_mask,
            stopping_criteria=stopping_criteria,
        )
        return outputs

    @property
    def config(self):
        return self.language_model.config
