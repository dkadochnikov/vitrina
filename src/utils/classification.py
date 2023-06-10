from torch import nn
from torch.nn import CTCLoss

from src.utils.common import PositionalEncoding, compute_ctc_loss


MAX_COLOUR = 255
AVER_LETTER_WIDTH = 6


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
        ocr=None,
        char2int: dict = None,
        alpha: float = 1,
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
        self.ocr = ocr
        self.ctc_criterion = CTCLoss(reduction="sum", zero_infinity=True)
        self.char2int = char2int
        self.alpha = alpha
        self.letter_count = width // AVER_LETTER_WIDTH
        self.linear = nn.Linear(emb_size, 2 * self.letter_count * height)

    def forward(self, input_batch):
        batch_size, slice_count, height, width = input_batch["slices"].shape
        slices = input_batch["slices"].view(batch_size, slice_count, height * width).clone()

        slices /= MAX_COLOUR
        slices = self.emb(slices)
        slices = self.positional(slices).permute(1, 0, 2)

        encoder_output = self.encoder(slices, src_key_padding_mask=input_batch["attention_mask"]).permute(1, 0, 2)
        encoder_output_drop = self.dropout(encoder_output)
        embeddings = encoder_output_drop.mean(dim=1)

        logits = self.classifier(embeddings)
        loss = self.criterion(logits, input_batch["labels"].unsqueeze(1))
        result = {"loss": loss, "logits": logits}

        if self.ocr:
            encoder_output = self.linear(encoder_output[input_batch["attention_mask"] == 1])
            seq_len = encoder_output.shape[0]
            encoder_output = encoder_output.view(seq_len, 1, height, 2 * self.letter_count)

            result["ce_loss"] = result["loss"].clone()
            result["ctc_loss"] = compute_ctc_loss(
                self.ctc_criterion, self.ocr, encoder_output, input_batch["texts"], self.char2int
            )
            result["loss"] = result["ce_loss"] + self.alpha * result["ctc_loss"]

        return result
