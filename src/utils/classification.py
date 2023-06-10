from torch import nn
from src.utils.common import PositionalEncoding


MAX_COLOUR = 255


class ToxicClassifier(nn.Module):
    def __init__(
            self,
            num_classes,
            pretrained=None,
            emb_size=768,
            n_head=12,
            n_layers=8,
            height=16,
            width=32,
    ):
        super().__init__()
        self.classifier = nn.Linear(emb_size, num_classes - 1)
        if pretrained:
            self.encoder = pretrained.encoder
            self.emb = pretrained.emb
            self.positional = pretrained.positional_enc
        else:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, dim_feedforward=emb_size),
                num_layers=n_layers,
            )
            self.emb = nn.Linear(height * width, emb_size, bias=False)
            self.positional = PositionalEncoding(emb_size)
        self.dropout = nn.Dropout(0.1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes

    def forward(self, input_batch):
        batch_size, slice_count, height, width = input_batch["slices"].shape
        slices = input_batch["slices"].view(batch_size, slice_count, height * width).clone()

        slices /= MAX_COLOUR
        slices = self.emb(slices)
        slices = self.positional(slices).permute(1, 0, 2)

        encoder_output = self.encoder(slices, src_key_padding_mask=input_batch["attention_mask"]).permute(1, 0, 2)
        encoder_output = self.dropout(encoder_output)
        embeddings = encoder_output.mean(dim=1)

        logits = self.classifier(embeddings)
        loss = self.criterion(logits, input_batch["labels"].unsqueeze(1))

        return {"loss": loss, "logits": logits}
