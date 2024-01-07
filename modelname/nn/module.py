from typing import Dict, Tuple, List, Type
from torch import Tensor
import torch
from e3nn.o3 import Irrep
from e3nn.o3 import Irreps
from e3nn.o3 import TensorProduct
from e3nn.nn import Gate
from e3nn.nn import Activation

from .from_equiformer import LinearRS
from .from_equiformer import TensorProductRescale
from .from_equiformer import get_norm_layer


class SelfGateModule(torch.nn.Module):
    def __init__(self, irreps_in: str | Irreps, 
                 act_scalars, act_gates=None) -> None:
        """ act_scalars: Activation Function for Scalars (Irrep.l==0)\n
            act_gates: Activation Function for non-scalars (Irrep.l>0), 
            should be None if irreps_in only contain scalars."""
        super().__init__()
        assert irreps_in.dim is not 0
        self.irreps_in: Irreps = Irreps(irreps_in)
        
        self.irreps_x0: Irreps = Irreps("") # from self.irreps_in, only contains 0e or 0o
        self.irreps_x1: Irreps = Irreps("") # from self.irreps_in, do not contain 0e or 0o
        self.irreps_gates_x1: Irreps = Irreps("") # from dot product of self.irreps_x1, only contains 0e
        self.index_x0: List[int] = [] # get index list of scalar (l=0 elements) of 'x' parameter of forward function
        self.index_x1: List[int] = [] # get index list of non-scalar (l>0 elements) of 'x' parameter of forward function
        self.sort_irreps, self.p, self.inv = self.irreps_in.sort()
        flag: int = 0
        for i, (mul, ir) in enumerate(self.irreps_in):
            new_flag: int = flag + mul*ir.dim
            if ir.l == 0:
                # note that type(self.irreps_in[i]) == e3nn.o3._irreps._MulIr, not Irreps
                # self.irreps_x0 += Irreps(self.irreps_in[i]) will raise Error here
                self.irreps_x0 += Irreps(f"{mul}x{ir}") 
                self.index_x0 += list(range(flag, new_flag))  
            else:
                self.irreps_x1 += Irreps(f"{mul}x{ir}") 
                self.irreps_gates_x1 += Irreps(f"{mul}x0e")
                self.index_x1 += list(range(flag, new_flag)) 
            flag = new_flag
        assert len(self.index_x0) + len(self.index_x1) == self.irreps_in.dim
        del i, mul, ir, flag, new_flag

        # if self.is_scalar = True, then irreps only contains 0e or 0o, no L>0 Irrep; 
        #                   and act_gates should be None
        self.is_scalar: bool = None 
        # act_gates: for non-scalars (L>0)
        if self.irreps_x0 == self.irreps_in: # self.irreps_x1 == Irreps("")
            # self.irreps_in only contain scalars
            self.is_scalar = True
        else:
            # self.irreps_in contains non-scalars
            self.is_scalar = False
            assert act_gates is not None

        if self.is_scalar==True:
            # self.irreps_in only contain scalars
            self.scalar_gate = Activation(
                irreps_in=self.irreps_in,
                acts=[act_scalars for _ in self.irreps_in]
            )
        else:
            # self.irreps_in contains non-scalars
            self.dot_product_x1: torch.nn.Module = TensorProduct(
                irreps_in1=self.irreps_x1, 
                irreps_in2=self.irreps_x1, 
                irreps_out=self.irreps_gates_x1, 
                instructions=[(i, i, i, 'uuu', False)
                for i, (mul, ir) in enumerate(self.irreps_x1)]
            )
            self.gate = Gate(
                irreps_scalars=self.irreps_x0,
                act_scalars=[act_scalars for _ in self.irreps_x0],
                irreps_gates=self.irreps_gates_x1,
                act_gates=[act_gates for _ in self.irreps_x1],
                irreps_gated=self.irreps_x1
            )

    def forward(self, x: Tensor):
        """x.irreps: need not be simplified sorted irreps( 
        "10x0e+20x1o+30x2e", not "1x1e+9x0e+20x1o+30x2e", not "30x2e+10x0e+20x1o").
        "0e" or "0o" is not necessary for x.irreps.
        Note that x.shape=(num_batch, x.irreps.dim).

        Suppose x = torch.cat([x0, x1], dim=-1),
        The scalar part of x: x0(l==0) --> Activation(x0),
        The other part of x: x1(l>0) --> Activation(dot_product(x1, x1)) * x1
        """

        if self.is_scalar==True:
            # self.irreps_in only contain scalars
            out = self.scalar_gate(x)
        else:
            # self.irreps_in contains non-scalars
            # Order: scalars, gates, gated          (Notice !)
            x = torch.cat(
                [
                    x[..., self.index_x0], 
                    self.dot_product_x1(x[..., self.index_x1], x[..., self.index_x1]),
                    x[..., self.index_x1]
                    ], 
                dim=-1
            )
            x = self.gate(x)

            out = torch.empty_like(x)
            out[..., self.index_x0] = x[..., list(range(0, len(self.index_x0)))]
            out[..., self.index_x1] = x[..., list(range(len(self.index_x0), len(self.index_x0)+len(self.index_x1)))]

        return out




class MultiLayerPerceptionModule(torch.nn.Module):
    def __init__(self, irreps_list: List[Irreps], 
                 add_last_linear: str | None=None, 
                 if_act: bool=True,
                 if_norm: bool=True,
                 act_scalars=None, act_gates=None,
                 norm_type:str="layer"):
        r"""MLP Module: (Linear(rescale=`True`)->LayerNorm->Gate)->(Linear(rescale=`False`)..).

        irreps_list type:
            `[input-irreps, mid-irreps-1, mid-irreps-2, ..., output-irreps]`
        support dtype:
            `List[Irreps]`: support scalars and non-scalars. `[Irreps("2x2e"), Irreps("3x2e"), Irreps("2x2e")]`

        add_last_linear:
            `None`: No last_Linear
            `"no_bias"` (str): Add last_Linear with no bias 
            `"bias"` (str): Add last_Linear with bias 
            Notice that last-linear: from mid-irreps-last or input-irreps to output-irreps.
        
        norm_type: the parameter for function `get_norm_layer`.
            """
        super().__init__()
        assert len(irreps_list) >= 2 # begin, end
        self.irreps_list: List[Irreps] = irreps_list
        if add_last_linear==None:
            self.last_irreps: Irreps = None
        elif add_last_linear=="no_bias" or add_last_linear=="bias":
            self.last_irreps: Irreps = irreps_list.pop()
        else:
            raise ValueError(f"Unsupported Input add_last_linear.\n"
                    f"Notice that the Input add_last_linear: add_last_linear={add_last_linear}.")
        if if_act==True:
            assert act_scalars is not None

        modules = []
        input_channels = self.irreps_list[0]
        for i in range(1, len(self.irreps_list)):
            modules.append(
                LinearRS(
                    irreps_in=input_channels, 
                    irreps_out=self.irreps_list[i], 
                    bias=False,
                    rescale=True
                )
            )
            if if_norm==True:
                modules.append(
                    get_norm_layer(norm_type=norm_type)(
                        irreps=self.irreps_list[i], 
                        eps=1e-5, 
                        affine=True, 
                        normalization='component'
                    )
                )
            if if_act==True:
                modules.append(
                    SelfGateModule(
                        irreps_in=self.irreps_list[i], 
                        act_scalars=act_scalars, 
                        act_gates=act_gates
                    )
                )
            input_channels = self.irreps_list[i]

        if add_last_linear==None:
            pass
        elif add_last_linear=="no_bias":
            modules.append(
                LinearRS(
                    irreps_in=input_channels, 
                    irreps_out=self.last_irreps, 
                    bias=False,
                    rescale=False
                )
            )
        elif add_last_linear=="bias":
            modules.append(
                LinearRS(
                    irreps_in=input_channels, 
                    irreps_out=self.last_irreps, 
                    bias=True,
                    rescale=False
                )
            )
        
        self.net = torch.nn.Sequential(*modules)
           
    def forward(self, x: Tensor):
        x = self.net(x)
        return x




# DepthwiseTensorProduct in SOTA-Equiformer need further improvement necessarily.
# We need to pick out specific irreps_out from TensorProduct of irreps_in1 and irreps_in2,
#   for example: "1x1e"+"1x1e"--TensorProduct-->"1x0e+1x1e+1x2e"--pickout-->"1x1e"
# DepthwiseTensorProduct in SOTA-Equiformer do not pick out, but use LinearRS behind to pick out (SeparableFCTP).
# However, that method:"uvu"->Linear totally equal to FullyConnectedTensorProduct:"uvw" ! (Notice !)
# We can use FullyConnectedTensorProduct instead.
    
# In order to really acheve "uvu" TP, another DepthwiseTensorProductModule is developed here.
# According to "uvu" principle, irreps_out==irreps_in1,
#   for example: "1x1e"+"1x1e"--DepthwiseTensorProductModule-->"1x1e"
# In order to avoid bug, there must be "0e" in irreps_in2, 
#   for example: "1x0e"+"1x1e"--DepthwiseTensorProductModule-->"1x0e" (Bug!)
class DepthwiseTensorProductModule(torch.nn.Module):
    def __init__(self, irreps_in1: Irreps, irreps_in2: Irreps, 
        internal_weights: bool):
        r"""DepthwiseTensorProduct Module: `irreps_in1 x irreps_in2 --> irreps_out=irreps_in1`.

        `irreps_in2`:
            `Irrep("0e")` should be included.

        `internal_weights`:
            `True`: the weights of `DepthwiseTensorProductModule` will be generated automatically. 
        The parameter `weight` in the forward function is `None` ! 
        And the weight is the same for every edge,
        so `shared_weights=True`.
            `False`: the weights of `DepthwiseTensorProductModule` will be generated outside.
        The parameter `weight` in the forward function should not be `None`!  
        And the weight always comes from edge_length which is different for every edge,
        so `shared_weights=False`.

            """
        super().__init__()
        self.irreps_in1: Irreps = irreps_in1
        assert Irrep("0e") in irreps_in2
        self.irreps_in2: Irreps = irreps_in2
        self.irreps_out: Irreps = irreps_in1
        self.internal_weights: bool = internal_weights
        

        self.tp: torch.nn.Module = \
            TensorProductRescale(
                irreps_in1=self.irreps_in1, 
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out, 
                instructions=[(i_out, i_2, i_out, "uvu", True)
                                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                                for i_2, (_, ir_2) in enumerate(self.irreps_in2)
                                if ir_out in ir_out * ir_2
                            ],
                shared_weights=self.internal_weights, 
                internal_weights=self.internal_weights,
                bias=False,
                rescale=True,
                normalization=None
        )
        self.weight_numel: int = self.tp.tp.weight_numel # used in MLP for weight generation
           
    def forward(self, x_1: Tensor, x_2: Tensor, weight: Tensor=None):
        out: Tensor = self.tp(x=x_1, y=x_2, weight=weight)
        return out
   






