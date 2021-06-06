from src.dl.base import Regularization, Layer


class L2(Regularization):
    def update_gradients(self, layer: Layer) -> None:
        if not layer.gradients or not layer.weights:
            return
        (W, b), (dw, db) = layer.weights, layer.gradients
        dw += self._lamda * W / dw.shape[0]
        layer.set_gradients(dw, db)
