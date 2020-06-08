import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, pretrained_model, hidden_size = 768, question_length = 15, convolution = True, kernel_size = 3, padding = 1):
        super(Model, self).__init__()
        self.model = pretrained_model
        self.question_length = question_length
        self.conv = convolution

        if self.conv:
            self.start = nn.Conv1d(hidden_size, 1, kernel_size = kernel_size, padding = padding)
            self.end = nn.Conv1d(hidden_size, 1, kernel_size = kernel_size, padding = padding)
        else:
            self.start = nn.Linear(hidden_size, 1)
            self.end = nn.Linear(hidden_size, 1)

    def forward(self, input_tensor, attention_tensor, segment_tensor):
        out = self.model(input_tensor, attention_tensor, segment_tensor) # (batch_size, 512, 768)

        if self.conv:
            out = out.permute(0, 2, 1)[:, :, self.question_length:] # (batch_size, 768, length)
            out_start = self.start(out).squeeze(1) # (batch_size, length)
            out_end = selg.end(out).squeeze(1) # (batch_size, length)
        else:
            out_start = self.start(out).squeeze(2) # (batch_size, length)
            out_end = self.end(out).squeeze(2) # (batch_size, length)

        return (out_start, out_end)
