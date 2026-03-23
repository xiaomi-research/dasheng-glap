from torch import Tensor


class _TensorType:
    def __getitem__(self, shape: str) -> type:
        return Tensor


Float = _TensorType()
Int = _TensorType()
Bool = _TensorType()
