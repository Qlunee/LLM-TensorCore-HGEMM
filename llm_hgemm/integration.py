"""Frozen, bias-free FP16 Linear adapter; weights must not change."""
import torch
import torch.nn.functional as F
from torch import nn
from .ops import hgemm
from .dispatch import last_selected

class FrozenHgemmLinear(nn.Module):
    def __init__(self, linear, implementation="shape_auto"):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.bias is not None:
            raise ValueError("requires a bias-free nn.Linear")
        if not linear.weight.is_cuda or linear.weight.dtype != torch.float16:
            raise ValueError("construct after moving FP16 model to CUDA")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.implementation = implementation
        self.register_buffer("weight", linear.weight.detach())
        self.register_buffer("packed_weight",
                             linear.weight.detach().T.contiguous())
        self._weight_version = self._version(self.weight)
        self._packed_version = self._version(self.packed_weight)
        self.eval()

    @staticmethod
    def _version(tensor):
        return None if torch.is_inference(tensor) else tensor._version

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FrozenHgemmLinear is inference-only")
        if (self._version(self.weight) != self._weight_version
                or self._version(self.packed_weight) != self._packed_version):
            raise RuntimeError("weights changed: rebuild the adapter")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("invalid Linear input shape")
        if x.device != self.packed_weight.device or x.dtype != torch.float16:
            raise ValueError("input must be FP16 on the weight device")
        if not x.is_contiguous() or x.numel() == 0:
            self.last_backend = "torch"
            return F.linear(x, self.weight)
        y = hgemm(x.view(-1, self.in_features), self.packed_weight,
                  implementation=self.implementation)
        self.last_backend = (last_selected() if self.implementation == "shape_auto"
                             else self.implementation)
        return y.view(*x.shape[:-1], self.out_features)
