from typing import Dict, Tuple, List
from torch import Tensor
from e3nn.o3 import Irreps
from e3nn.o3 import Irrep
import torch

from ..utils.data import to_dtype_0e_scalar_irreps
from ..nn.from_equiformer import torch_geometric_2_0_3_glorot

class old_0_DotProductModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps], num_heads: None | int):
        r"""Only used for Irrep("0e") scalars.
        `num_heads`:
            `None`: input shape=`(num_batch, irreps_in.dim)`;
            `int`: input shape=`(num_batch, num_heads, irreps_in.dim)`.
        """
        super().__init__()
        self.irreps_in_dict: Dict[str, Irreps] = irreps_in_dict
        for mul, ir in self.irreps_in_dict.values():
            assert ir==Irrep("0e")
        self.num_heads: None | int = num_heads

        if self.num_heads is None:
            self.dot_product_parameters: torch.nn.ParameterDict = torch.nn.ParameterDict(
                {
                    f"DotProductParameters{k}": torch.nn.Parameter(
                        torch.randn(to_dtype_0e_scalar_irreps(irreps=v, dtype=int))
                        )
                    for k, v in self.irreps_in_dict.items()
                }
            ) 
        elif type(self.num_heads)==type(1): # isinstance(self.num_heads, int)
            self.dot_product_parameters: torch.nn.ParameterDict = torch.nn.ParameterDict(
                {
                    f"DotProductParameters{k}": torch.nn.Parameter(
                        torch.randn(self.num_heads, to_dtype_0e_scalar_irreps(irreps=v, dtype=int))
                        )
                    for k, v in self.irreps_in_dict.items()
                }
            ) 
        else:
            raise NotImplementedError

        self.dot_product_parameters = {
            k: torch_geometric_2_0_3_glorot(v)
            for k, v in self.dot_product_parameters.items()
        } # reinitialize as Equiformer


    def forward(self, x_dict: Dict[str, Tensor]):
        out_dict = {
            k: torch.sum(v * self.dot_product_parameters[f"DotProductParameters{k}"], dim=-1)
            for k, v in x_dict.items()
        } 
        # torch.einsum('bik, ik -> bi', v, ...) or torch.einsum('bk, k -> b', v, ...)
        # key: edge type, value shape=(num_edges, num_heads, irreps.dim) --> (num_edges, num_heads)
        return out_dict
