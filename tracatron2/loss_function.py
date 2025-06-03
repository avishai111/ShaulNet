from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        weights = targets[2] if len(targets) > 2 else None  # אופציונלי

        # ניתוק מהגרף החישובי
        mel_target = mel_target.detach()
        gate_target = gate_target.detach()

        mel_out, mel_out_postnet, gate_out, _ = model_output

        B = mel_out.size(0)  # גודל batch

        # -------- MEL LOSS -------- #
        mel_loss_1 = self.mse_loss(mel_out, mel_target).mean(dim=[1, 2])         # [B]
        mel_loss_2 = self.mse_loss(mel_out_postnet, mel_target).mean(dim=[1, 2]) # [B]
        mel_loss = mel_loss_1 + mel_loss_2  # [B]

        # -------- GATE LOSS -------- #
        # gate_out, gate_target בגודל [B, T]
        gate_loss = self.bce_loss(gate_out, gate_target).mean(dim=1)  # [B]

        # -------- שילוב עם משקולות -------- #
        if weights is not None:
            weights = weights.to(mel_loss.device)
            total_weight = weights.sum()
            weighted_loss = (mel_loss * weights).sum() + (gate_loss * weights).sum()
            return weighted_loss / total_weight
        else:
            return (mel_loss + gate_loss).mean()
