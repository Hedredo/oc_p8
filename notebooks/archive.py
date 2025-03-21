from enum import Enum


class BlockEfficientNetB3(Enum):
    BLOCK_7 = (346, 373)  # Plage des indices de couches pour le bloc 8
    BLOCK_6 = (258, 345)  # Plage des indices de couches pour le bloc 7
    BLOCK_5 = (185, 257)  # Plage des indices de couches pour le bloc 6
    BLOCK_4 = (112, 184)  # Plage des indices de couches pour le bloc 5
    BLOCK_3 = (69, 111)  # Plage des indices de couches pour le bloc 4
    BLOCK_2 = (26, 68)  # Plage des indices de couches pour le bloc 3
    BLOCK_1 = (4, 25)  # Plage des indices de couches pour le bloc 2
    # Ajouter d'autres blocs si nécessaire

    def get_layer_range(self):
        return self.value
    
class BlockResnet50(Enum):
    STAGE_4 = (154, 186)  # Plage des indices de couches pour le bloc 8
    STAGE_3 = (87, 153)  # Plage des indices de couches pour le bloc 7
    STAGE_2 = (42, 86)  # Plage des indices de couches pour le bloc 6
    STAGE_1 = (8, 41)  # Plage des indices de couches pour le bloc 5

    def get_layer_range(self):
        return self.value
    

class ProgressiveUnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        enums,
        monitor="val_MeanIoU",
        min_patience=1,
        patience_mult=3,
        threshold=0.01,
        min_lr=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = min_patience
        self.patience_mult = patience_mult
        self.threshold = threshold
        self.min_lr = min_lr
        self.enums = list(enums)
        self.best_metric = None
        self.wait = 0
        self.level = 0
        self.adaptative_lr = None

    def on_train_begin(self, logs=None):
        """Initialisation correcte de l'adaptive learning rate au début de l'entraînement."""
        initial_lr = float(
            self.model.optimizer.learning_rate.numpy()
        )  # Convertir en scalaire
        self.adaptative_lr = pow(self.min_lr / initial_lr, 1 / len(self.enums))

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if current_metric is None:
            print(
                f"Avertissement : '{self.monitor}' est absent des logs pour l'epoch {epoch}."
            )
            return  # Ne fait rien et attend l'epoch suivant

        # Si c'est le premier epoch, assigner la métrique actuelle comme la meilleure métrique et attendre l'epoch suivant
        if self.best_metric is None:
            self.best_metric = current_metric
            return

        # Vérifie si la métrique n'a pas changé suffisamment
        if abs(self.best_metric - current_metric) < self.threshold:
            self.wait += 1
        else:
            self.wait = 0

        # Si la métrique n'a pas évolué suffisamment pendant 'patience' époques, on dégèle des couches
        if self.wait >= self.patience:
            if self.level < len(self.enums):
                self.wait = 0
                self.best_metric = current_metric
                self.unfreeze_layers_in_model()
                self.level += 1
                if self.patience.mult > 1:
                    self.patience *= self.patience_mult
                    self.patience_mult -= 1
            # Si toutes les couches ont été degelées, on continue l'entraînement sans rien changer
            else:
                return

    def unfreeze_layers_in_model(self):
        # Récupérer les indices des couches à dégeler en fonction du bloc
        start_layer, end_layer = self.enums[self.level].get_layer_range()

        # Récupérer les couches du modèle en fonction des indices
        layers_to_unfreeze = self.model.layers[start_layer:end_layer]

        # Dégeler les couches sélectionnées
        for layer in layers_to_unfreeze:
            layer.trainable = True

        # Leaning rate decreasing
        new_lr = self.model.optimizer.learning_rate.numpy() * self.adaptative_lr
        self.model.optimizer.learning_rate.assign(new_lr)

        # Force le modèle à prendre en compte les changements de couches
        self.model.make_train_function(force=True)
