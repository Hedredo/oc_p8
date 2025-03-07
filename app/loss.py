# Create a class from tf.keras.losses.Loss combining dice loss and focal loss
class DiceFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        num_classes=8,
        # from_logits=False,
        smooth=1e-6,
        alpha=0.8,
        gamma=2.0,
        name="dice_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # self.from_logits = from_logits
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Compute the dice loss
        dice_loss = sm.losses.DiceLoss(
            beta=1,
            class_weights=None,
            class_indexes=[*range(self.num_classes)],
            smooth=self.smooth,
        )(y_true, y_pred)
        # Compute the focal loss
        focal_loss = sm.losses.CategoricalFocalLoss(
            alpha=0.25,
            gamma=2.0,
            class_indexes=[*range(self.num_classes)],
        )(y_true, y_pred)
        # Return the sum of both losses
        loss = self.alpha * dice_loss + (1 - self.alpha) * focal_loss
        return loss
