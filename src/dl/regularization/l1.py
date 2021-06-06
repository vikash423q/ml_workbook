from src.dl.base import Regularization, Layer


class L1(Regularization):
    def update_gradients(self, layer: Layer) -> None:
        if not layer.gradients:
            return
        dw, db = layer.gradients
        dw += self._lamda / (2 * dw.shape[0])
        layer.set_gradients(dw, db)
