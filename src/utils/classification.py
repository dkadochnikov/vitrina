from torch import nn


MAX_COLOUR = 255


class ToxicClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, pretrained):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes - 1)
        self.encoder = pretrained.encoder
        self.emb = pretrained.emb
        self.positional = pretrained.positional_enc
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
