from typing import Dict, Tuple, List, Set, Union
from torch_geometric.data import HeteroData
from e3nn.o3 import Irreps

import torch
import torch.nn as nn

from e3nn.o3 import TensorProduct
from torch.nn.functional import one_hot
from e3nn.o3 import Linear
from e3nn.nn import Gate
# from e3nn.o3 import SphericalHarmonics

from .graph import to_dtype_edge


"""
MIT License

Copyright (c) 2023 Xiaoxun-Gong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[..., None] - offset
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[..., None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianBasis(nn.Module):
    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianBasis, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.Tensor((offset[1] - offset[0]) * torch.ones_like(offset)) # FloatTensor
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )
    









# ori=orientation, FullyConnectedTensorProduct
class OriEmbeddingBlock(nn.Module): 
    def __init__(self, ori_irreps: Union[str, Irreps], in_irreps: Union[str, Irreps]= "1o") -> None:
        super().__init__()
        irreps_in1: Irreps = Irreps(in_irreps)
        irreps_in2: Irreps = Irreps(in_irreps)
        irreps_out: Irreps = Irreps(ori_irreps)

        self.weights: nn.Parameter = nn.Parameter(
            torch.cat([torch.randn(mul_1*mul_2*mul_out)
                        for _, (mul_1, ir_1) in enumerate(irreps_in1)
                        for _, (mul_2, ir_2) in enumerate(irreps_in2)
                        for _, (mul_out, ir_out) in enumerate(irreps_out)
                        if ir_out in ir_1 * ir_2
                        ], 
                    dim=-1
                    )
        ) # no ".reshape(1, -1)" because of shared_weights=True

        self.tensor_product: nn.Module = \
            TensorProduct(irreps_in1, 
                        irreps_in2,
                        irreps_out, 
                        instructions=[(i_1, i_2, i_out, "uvw", True)
                                        for i_1, (_, ir_1) in enumerate(irreps_in1)
                                        for i_2, (_, ir_2) in enumerate(irreps_in2)
                                        for i_out, (_, ir_out) in enumerate(irreps_out)
                                        if ir_out in ir_1 * ir_2
                                    ],
                        shared_weights=True,
                        internal_weights=False)


    def forward(self, ori):
        out = self.tensor_product(ori, ori, self.weights)
        return out
    





class SelfGate(nn.Module):
    def __init__(self, irreps_in: Union[str, Irreps], 
                 act_scalars, act_gates) -> None:
        super().__init__()
        irreps_in: Irreps = Irreps(irreps_in)
        # assert len(irreps_in) > 0

        irreps_x0: Irreps = Irreps("") # from irreps_in, only contains 0e or 0o
        irreps_x1: Irreps = Irreps("") # from irreps_in, do not contain 0e or 0o
        irreps_gates_x1: Irreps = Irreps("") # from dot product of irreps_x1, only contains 0e
        self.index_end_x0: int = 0 # get total number of l=0 elements of 'x' parameter of forward function
        for i, (mul, ir) in enumerate(irreps_in):
            if ir.l == 0:
                # note that type(irreps_in[i]) == e3nn.o3._irreps._MulIr, not Irreps
                # irreps_x0 += Irreps(irreps_in[i]) will raise Error here
                irreps_x0 += Irreps(f"{mul}x{ir}") 
                self.index_end_x0 += mul # suppose 'x' parameter of forward function is sorted and simplified by irreps
            else:
                irreps_x1 += Irreps(f"{mul}x{ir}") 
                irreps_gates_x1 += Irreps(f"{mul}x0e")
        del i, mul, ir

        self.dot_product_x1: nn.Module = TensorProduct(
            irreps_in1=irreps_x1, 
            irreps_in2=irreps_x1, 
            irreps_out=irreps_gates_x1, 
            instructions=[(i, i, i, 'uuu', False)
            for i, (mul, ir) in enumerate(irreps_x1)]
        )

        self.gate = Gate(
            irreps_scalars=irreps_x0,
            act_scalars=[act_scalars for _ in irreps_x0],
            irreps_gates=irreps_gates_x1.simplify(),
            act_gates=[act_gates], # [lambda x: x], 
            irreps_gated=irreps_x1
        )

    def forward(self, block_others_list: List):
        """block.irreps: should be simplified sorted irreps. 
        "10x0e+20x1o+30x2e", not "1x1e+9x0e+20x1o+30x2e", not "30x2e+10x0e+20x1o".
        "0e" or "0o" is not necessary for block.irreps.
        Note that block.shape=(num_blocks, simple_sorted_block_irreps).

        Suppose block = torch.cat([x0, x1], dim=-1),
        The scalar part of block: x0(l==0) --> Activation(x0),
        The other part of block: x1(l>0) --> Activation(dot_product(x1, x1)) * x1
        """
        block = block_others_list[0]
        others = block_others_list[1]
        del block_others_list

        block = torch.cat(
            [
                block[:, :self.index_end_x0], # scalar
                self.dot_product_x1(block[:, self.index_end_x0:], block[:, self.index_end_x0:]), # gates
                block[:, self.index_end_x0:], # gated
                ], 
            dim=-1
        )
        block = self.gate(block)
        # can pass single-parameter to BlockUpdateBlock directly
        return [block, others]






class BlockUpdateBlock(nn.Module):
    def __init__(self, block_irreps: Union[str, Irreps], others_irreps: Union[str, Irreps]) -> None:
        super().__init__()
        irreps_in1: Irreps = Irreps(block_irreps)
        irreps_in2: Irreps = Irreps(others_irreps)
        irreps_out: Irreps = Irreps(block_irreps)

        self.weights: nn.Parameter = nn.Parameter(
            torch.cat([torch.randn(mul_out*mul_2)
                        for _, (mul_out, ir_out) in enumerate(irreps_out)
                        for _, (mul_2, ir_2) in enumerate(irreps_in2)
                        if ir_out in ir_out * ir_2
                        ], 
                    dim=-1
                    )
        )# no ".reshape(1, -1)" because of shared_weights=True

        self.tensor_product: nn.Module = \
            TensorProduct(irreps_in1, 
                        irreps_in2,
                        irreps_out, 
                        instructions=[(i_out, i_2, i_out, "uvu", True)
                                        for i_out, (_, ir_out) in enumerate(irreps_out)
                                        for i_2, (_, ir_2) in enumerate(irreps_in2)
                                        if ir_out in ir_out * ir_2
                                    ],
                        shared_weights=True,
                        internal_weights=False)
            
    def forward(self, block_others_list: List):
        # It is recommend to have single-parameter, because of nn.Sequential.
        # the code def forward(self, block, others): will raise error
        #   Sequential.forward() takes 2 positional arguments but 3 were given
        block = block_others_list[0]
        others = block_others_list[1]
        del block_others_list
        # there should be no in_place operation in forward function!
        # the code block += self.tensor_product(x=block, y=others, weight=self.weights) will raise error
        #   RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation 
        block = block + self.tensor_product(x=block, y=others, weight=self.weights)
        # others also may be changed
        return [block, others]






NUM_ONE_HOT: int = 118 # number of total elements of periodic table
CUTOFF_RADIUS: int = 7.2 # Angstrom?TODO
NUM_GAUSSIANS: int = 128

ELEMENT_IRREPS: Irreps = Irreps("8x0e")
DISTANCE_IRREPS: Irreps = Irreps(f"16x0e")
ORIENTATION_IRREPS: Irreps = Irreps(f"16x1e+4x2e")
#TODO OTHERS_IRREPS: Irreps = (2*ELEMENT_IRREPS+DISTANCE_IRREPS+ORIENTATION_IRREPS).simplify() 
#should be in order # OTHERS_IRREPS=32x0e+16x1e+4x2e
OTHERS_IRREPS: Irreps = (DISTANCE_IRREPS+ORIENTATION_IRREPS).simplify() 

ACT_SCALARS = torch.relu
ACT_GATES = torch.nn.functional.sigmoid

class Net(nn.Module):
    def __init__(self, block_irreps_dict: Dict[Tuple[int, int], Irreps], 
                 ): # irreps_embed_node: Irreps
        super().__init__()
        self.init_model_input_for_debug = block_irreps_dict # For Reproduce/Debug, Not for train
        # the code set([edge for edge in block_irreps_dict.keys()]) will be wrong
        # the code set(edge for edge in block_irreps_dict.keys()) will still be wrong
        # node_types: Set[Tuple[int, int]] = {(83, 83)}
        # corrected: node_types: Set[Tuple[int, int]] = {83}
        node_types: Set[int] = set([])
        for edge in block_irreps_dict.keys():
            node_types.update(edge)

        self.node_embedding_dict = nn.ModuleDict()
        for node in node_types: # node int
            self.node_embedding_dict[str(node)] = \
                Linear(irreps_in=f"{NUM_ONE_HOT}x0e", irreps_out=ELEMENT_IRREPS)

        self.distance_embedding_dict = nn.ModuleDict()
        self.orientation_embedding_dict = nn.ModuleDict()
        for edge in block_irreps_dict.keys(): # edge (83, 83) to "83-83"
            edge: str = to_dtype_edge(edge, dtype=str)
            # the code self.distance_embedding_dict[begin, edge, end] will raise error:
            #   module name should be a string. Got tuple
            self.distance_embedding_dict[edge] = \
                nn.Sequential(
                    GaussianBasis(start=0.0, stop=CUTOFF_RADIUS, n_gaussians=NUM_GAUSSIANS, trainable=False),
                    Linear(irreps_in=f"{NUM_GAUSSIANS}x0e", irreps_out=DISTANCE_IRREPS)
                )
            self.orientation_embedding_dict[edge] = \
                OriEmbeddingBlock(ori_irreps=ORIENTATION_IRREPS) 
                # not SphericalHarmonics here because we need all-parity=1-vector in BlockUpdateBlock

        self.block_update_dict = nn.ModuleDict()
        for edge, block_irreps in block_irreps_dict.items(): # edge (83, 83) to "83-83"
            edge: str = to_dtype_edge(edge, dtype=str)
            self.block_update_dict[edge] = \
                nn.Sequential(
                    BlockUpdateBlock(block_irreps=block_irreps, others_irreps=OTHERS_IRREPS), 
                    SelfGate(irreps_in=block_irreps, act_scalars=ACT_SCALARS, act_gates=ACT_GATES),
                    BlockUpdateBlock(block_irreps=block_irreps, others_irreps=OTHERS_IRREPS)
                )
                

    
    def forward(self, data: HeteroData):
        
        distance_dict = {}
        orientation_dict = {}
        block_dict = {}
        for edge, edge_fea in data.edge_fea_dict.items(): # edge ("83", "83-83", "83") to "83", "83-83", "83"
            begin, edge, end = edge

            dist = edge_fea[:, 0] # distance
            ori = edge_fea[:, 1:4] # orientation
            block = edge_fea[:, 4:] # overlaps block

            distance_dict[edge] = self.distance_embedding_dict[edge](dist)
            orientation_dict[edge] = self.orientation_embedding_dict[edge](ori)
            block_dict[edge] = block
            del begin, end, edge
        
        element_dict = {}
        for node, atomic_number in data.x_dict.items(): # node int
            # the code raise element_dict[str(node)] = one_hot(atomic_number, num_classes=NUM_ONE_HOT) 
            # will raise error later, because torch.int64 in element_dict but torch.float.. in default
            #   RuntimeError: both inputs should have same dtype
            # it's useless to add the code element_dict[str(node)].type(torch.get_default_dtype())
            #   or  element_dict[str(node)].to(torch.get_default_dtype())
            #Correct but not good element_dict[str(node)] = one_hot(atomic_number, num_classes=NUM_ONE_HOT).type(torch.get_default_dtype())
            element_dict[str(node)] = \
                dist.new_tensor(one_hot(atomic_number, num_classes=NUM_ONE_HOT))
            element_dict[str(node)] = self.node_embedding_dict[str(node)](element_dict[str(node)])
            del node

        
        for edge, block in block_dict.items():
            #################################################critical part 
            #TODO note that OTHERS_IRREPS: Irreps = (2*ELEMENT_IRREPS+DISTANCE_IRREPS+ORIENTATION_IRREPS).simplify()
            # note that OTHERS_IRREPS: Irreps = (DISTANCE_IRREPS+ORIENTATION_IRREPS).simplify()
            '''TODO
            begin: str = edge.split("-")[0]
            end: str = edge.split("-")[1]
            others = torch.cat(
                [element_dict[begin], element_dict[end], distance_dict[edge], orientation_dict[edge]],
                dim=-1
                ) '''
            others = torch.cat(
                [distance_dict[edge], orientation_dict[edge]],
                dim=-1
                ) 
            # It is recommend to have single-parameter, because of nn.Sequential.
            # the code block_dict[edge] = self.block_update_dict[edge](block, others) will raise error
            #   Sequential.forward() takes 2 positional arguments but 3 were given
            block_dict[edge], others = self.block_update_dict[edge]([block, others])
            del block, others, edge# begin, end
            

        return block_dict


