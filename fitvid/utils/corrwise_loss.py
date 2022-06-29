from corrwise.corr_wise import CorrWise


class CorrWiseLoss(CorrWise):
    # Version of CorrWiseLoss which allows videos to be passed in instead of just images
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target):
        input_shape = pred.shape
        if len(input_shape) == 5: # video
            # reshape pred to be [batch*time, channel, height, width]
            pred = pred.reshape(-1, *input_shape[2:])
            target = target.reshape(-1, *input_shape[2:])
        return super().forward(pred, target)
