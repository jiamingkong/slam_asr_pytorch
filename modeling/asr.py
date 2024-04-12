"""
The main body of the ASR model,

User: <Speech> <Prompt>
Model: <Transcription>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASR(nn.Module):
    def __init__(self, speech_encoder_model_id, language_model_id, downsample_K = 5, hidden_dim = 2048, train_mode="adapter"):
        assert train_mode in ["adapter", "full"]
        super(ASR, self).__init__()
        
        self.language_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=args.trust_remote_code,
        )
        # project_dim = self.language_model's embedding size
        project_dim = self.language_model.config.hidden_size

        self.speech_encoder = SpeechEncoder(speech_encoder_model_id, project_dim, downsample_K = downsample_K, hidden_dim=hidden_dim, train_mode=train_mode)
        self.set_gradient(train_mode)

    def set_gradient(self, train_mode):
        # freeze the whole language_model
        for param in self.language_model.parameters():
            param.requires_grad = False

        # call set_gradient for speech encoder
        self.speech_encoder.set_gradient(train_mode)

    def forward(self, input_dict):