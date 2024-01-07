from typing import Dict, Tuple, List
from torch import Tensor
from e3nn.o3 import Irreps
from e3nn.o3 import Irrep
from torch_scatter import scatter
from torch_geometric.utils import softmax 
import torch

from .from_equiformer import LinearRS
from .from_equiformer import Vec2AttnHeads
from .from_equiformer import AttnHeads2Vec
from .module import SelfGateModule
from .module import MultiLayerPerceptionModule
from .module import DepthwiseTensorProductModule
from ..utils.data import to_dtype_0e_scalar_irreps
from ..utils.data import to_dtype_nodes_from_edge
from ..utils.data import divide_irreps
from ..utils.data import if_all_include_irrep
from ..utils.data import if_only_contain_irrep
from ..utils.from_equiformer import get_mul_0

# TODO
# search for *[1 for _ in range(   and wrap the code by one function


class LinearModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps], irreps_out_dict: Dict[str, Irreps], bias: bool):
        """
            Notice that non-scalars have no bias even `bias=True`.

            Notice that keys of `irreps_in_dict` should be the same as the keys of `irreps_out_dict` strictly.

        """
        super().__init__()
        self.irreps_in_dict: Dict[str, Irreps] = irreps_in_dict
        self.irreps_out_dict: Dict[str, Irreps] = irreps_out_dict
        self.bias: bool = bias

        self.linear_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
                f"Linear({k})": 
                LinearRS(irreps_in=v, irreps_out=self.irreps_out_dict[k], bias=self.bias, rescale=True)
                for k, v in self.irreps_in_dict.items()
            }
        )

    def forward(self, x_dict: Dict[str, Tensor]):
        out_dict: Dict[str, Tensor] = {
            k: self.linear_dict[f"Linear({k})"](v)
            for k, v in x_dict.items()
        }
        return out_dict




class SelfGateModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps], act_scalars, act_gates=None):
        """
            Notice that all SelfGate Module in `SelfGateModuleDict` use the same `act_scalars` and `act_gates`.
        
            Notice that if all Irreps in `irreps_in_dict` are scalars ("0e"), then `act_gates` is not needed.
        """
        super().__init__()
        self.irreps_in_dict: Dict[str, Irreps] = irreps_in_dict
        self.act_scalars = act_scalars
        self.act_gates = act_gates

        if self.act_gates is None:
            assert if_only_contain_irrep(target=irreps_in_dict, irrep=Irrep("0e"))

        self.gate_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
                f"SelfGate({k})": 
                SelfGateModule(irreps_in=v, act_scalars=self.act_scalars, act_gates=self.act_gates)
                for k, v in self.irreps_in_dict.items()
            }
        )

    def forward(self, x_dict: Dict[str, Tensor]):
        """
            Notice that Irreps vector in x_dict donnot need to be simplified.
            """
        out_dict: Dict[str, Tensor] = {
            k: self.gate_dict[f"SelfGate({k})"](v)
            for k, v in x_dict.items()
        }
        return out_dict




class MultiLayerPerceptionModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps], 
                 irreps_out_dict: Dict[str, Irreps], 
                 irreps_mid_list: List[Irreps]=[], 
                 add_last_linear: str | None=None, 
                 if_act: bool=True,
                 if_norm: bool=True,
                 act_scalars=None, act_gates=None,
                 norm_type:str="layer"):
        r"""
            Notice that mlp dictionary (`self.mlp_dict`) is built corresponding to
        each key-value in `irreps_in_dict` or `irreps_out_dict` strictly. 
            Notice that input parameter `irreps_list` for mlp is stored in `self.irreps_list_dict`,
        which satisfies `self.irreps_list_dict[key]` = `irreps_in_dict[key] + irreps_mid_list + irreps_out_dict[key]`.
        
        From MultiLayerPerceptionModule
        ---

        MLP Module: (Linear->LayerNorm->Gate)->(Linear..).

        irreps_list type:
            `[input-irreps, mid-irreps-1, mid-irreps-2, ..., output-irreps]`
        support dtype:
            `List[Irreps]`: support scalars and non-scalars. `[Irreps("2x2e"), Irreps("3x2e"), Irreps("2x2e")]`

        add_last_linear:
            `None`: No last_Linear
            `"no_bias"` (str): Add last_Linear with no bias
            `"bias"` (str): Add last_Linear with bias 
        
        norm_type: the parameter for function `get_norm_layer`.
            """
        super().__init__()
        self.irreps_in_dict: Dict[str, Irreps] = irreps_in_dict
        self.irreps_out_dict: Dict[str, Irreps] = irreps_out_dict
        assert Irreps("") not in irreps_mid_list, "lost all data in the middle"
        assert None not in irreps_mid_list

        self.irreps_list_dict: Dict[str, List[Irreps]] = {
            k: [v] + irreps_mid_list + [self.irreps_out_dict[k]]
            for k, v in self.irreps_in_dict.items()
        }
        self.mlp_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
                f"MultiLayerPerceptionModule({k})": 
                MultiLayerPerceptionModule(
                    irreps_list=self.irreps_list_dict[k],
                    add_last_linear=add_last_linear, if_act=if_act, if_norm=if_norm,
                    act_scalars=act_scalars, act_gates=act_gates, norm_type=norm_type
                )
                for k, v in self.irreps_in_dict.items()
            }
        )
           
    def forward(self, x_dict: Dict[str, Tensor]):
        out_dict: Dict[str, Tensor] = {
            k: self.mlp_dict[f"MultiLayerPerceptionModule({k})"](v)
            for k, v in x_dict.items()
        }
        return out_dict




class DepthwiseTensorProductModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps], 
                 irreps_edge_vec_embed: Irreps, 
                 dtp_internal_weights: bool,
                 irreps_edge_length_embed: Irreps=None,
                 mlp_irreps_mid_list: List[Irreps]=None, 
                 mlp_add_last_linear: str | None=None, 
                 mlp_if_act: bool=None,
                 mlp_if_norm: bool=None,
                 mlp_act_scalars=None, mlp_act_gates=None,
                 mlp_norm_type:str=None):
        r"""DepthwiseTensorProductModuleDict: for each key in `irreps_in_dict`,
        `irreps_in_dict[key] x irreps_edge_vec_dict[key] --> irreps_out=irreps_in_dict[key]`

        `irreps_edge_length_dict`, `mlp_irreps_mid_list`:
            Only contain `Irrep("0e")`.
        `irreps_edge_vec_dict`:
            Include `Irrep("0e")`.

        `internal_weights`:
            `True`: the weights of `DepthwiseTensorProductModule` will be generated automatically. 
        And the weight is the same for every edge. MLP is disabled in the case.
        So `irreps_edge_length_dict` and `mlp-parameters` need not be provided (`None`).
            `False`: the weights of `DepthwiseTensorProductModule` will be generated by MLP 
        from `edge_length_dict`.
        And the weight always comes from edge_length which is different for every edge.
        So `irreps_edge_length_dict` and `mlp-parameters` should be provided (not be `None`).


        From MultiLayerPerceptionModule
        ---

        MLP Module: (Linear->LayerNorm->Gate)->(Linear..).

        irreps_list type:
            `[input-irreps, mid-irreps-1, mid-irreps-2, ..., output-irreps]`
        support dtype:
            `List[Irreps]`: support scalars and non-scalars. `[Irreps("2x2e"), Irreps("3x2e"), Irreps("2x2e")]`

        add_last_linear:
            `None`: No last_Linear
            `"no_bias"` (str): Add last_Linear with no bias
            `"bias"` (str): Add last_Linear with bias 
        
        norm_type: the parameter for function `get_norm_layer`.
        """
        super().__init__()
        self.irreps_in1_dict: Dict[str, Irreps] = irreps_in_dict
        self.irreps_out_dict: Dict[str, Irreps] = irreps_in_dict
        assert if_all_include_irrep(target=irreps_edge_vec_embed, irrep=Irrep("0e")), \
            f"Error: Not All Values in irreps_edge_vec_dict include Irrep(\"0e\")!"
        self.irreps_in2: Irreps = irreps_edge_vec_embed
        self.dtp_internal_weights: bool = dtp_internal_weights

        self.dtp_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
                f"DepthwiseTensorProductModule({k})": 
                DepthwiseTensorProductModule(
                    irreps_in1=v,
                    irreps_in2=self.irreps_in2,
                    internal_weights=self.dtp_internal_weights
                )

                for k, v in self.irreps_in1_dict.items()
            }
        )

        if self.dtp_internal_weights is False:
            assert if_only_contain_irrep(target=irreps_edge_length_embed, irrep=Irrep("0e"))
            assert if_only_contain_irrep(target=mlp_irreps_mid_list, irrep=Irrep("0e"))
            self.irreps_edge_length_embed: Irreps = irreps_edge_length_embed
            self.mlp_dict: torch.nn.Module = MultiLayerPerceptionModuleDict(
                irreps_in_dict={k: self.irreps_edge_length_embed for k in self.irreps_in1_dict.keys()},
                irreps_out_dict={
                    k: to_dtype_0e_scalar_irreps(
                        irreps=self.dtp_dict[f"DepthwiseTensorProductModule({k})"].weight_numel,
                        dtype=Irreps
                    ) 
                    for k in self.irreps_in1_dict.keys()
                },
                irreps_mid_list=mlp_irreps_mid_list,
                add_last_linear=mlp_add_last_linear, if_act=mlp_if_act, if_norm=mlp_if_norm,
                act_scalars=mlp_act_scalars, act_gates=mlp_act_gates, norm_type=mlp_norm_type
            )


    def forward(self, edge_fea_dict: Dict[str, Tensor], edge_vec_dict: Dict[str, Tensor], edge_length_dict: Dict[str, Tensor]=None):
        # edge_fea: (num_edges, irreps_edge.dim) Or (num_edges, num_heads, irreps_edge.dim)...
        # edge_vec, edge_length: (num_edges, irreps_xx.dim)
        # We need to expand edge_vec, edge_length to avoid Tensor shape conflict.
        edge_vec_dict = {
            k: v.view(
                v.shape[0], *[1 for _ in range(edge_fea_dict[k].dim()-2)], v.shape[1]
                ).expand([*edge_fea_dict[k].shape[:-1], v.shape[1]])
            for k, v in edge_vec_dict.items()
        }
        edge_length_dict = {
            k: v.view(
                v.shape[0], *[1 for _ in range(edge_fea_dict[k].dim()-2)], v.shape[1]
                ).expand([*edge_fea_dict[k].shape[:-1], v.shape[1]])
            for k, v in edge_length_dict.items()
        }
        
        if self.dtp_internal_weights is False:
            assert edge_length_dict is not None, "Lost Parameters: edge_length_dict in forward function of DepthwiseTensorProductModuleDict."
            weight_dict: Dict[str, Tensor] = self.mlp_dict(edge_length_dict)
        else:
            weight_dict: Dict[str, Tensor] = {
                k: None
                for k, v in edge_fea_dict.items()
            }

        out_dict: Dict[str, Tensor] = {
            k: self.dtp_dict[f"DepthwiseTensorProductModule({k})"](v, edge_vec_dict[k], weight_dict[k])
            for k, v in edge_fea_dict.items()
        }
        return out_dict




class Vec2AttnHeadsModuleDict(torch.nn.Module):
    def __init__(self, irreps_head_dict: Dict[str, Irreps], 
                 num_head: int):
        r"""Reshape vectors of shape `[N, num_heads*irreps_head_dict[key]]` to vectors of shape
        `[N, num_heads, irreps_head_dict[key]]` for each key in `irreps_head_dict`.
        """
        super().__init__()
        self.irreps_head_dict: Dict[str, Irreps] = irreps_head_dict
        self.num_head: int = num_head

        self.vec2heads_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
            f"Vec2AttnHeads({k})": Vec2AttnHeads(irreps_head=v, num_heads=self.num_head)
            for k, v in self.irreps_head_dict.items()
            }
        )


    def forward(self, x_dict: Dict[str, Tensor]):
        out_dict: Dict[str, Tensor] = {
            k: self.vec2heads_dict[f"Vec2AttnHeads({k})"](v)
            for k, v in x_dict.items()
        }
        return out_dict




class AttnHeads2VecModuleDict(torch.nn.Module):
    def __init__(self, irreps_head_dict: Dict[str, Irreps]):
        r"""Reshape vectors of shape `[N, num_heads, irreps_head_dict[key]]` to vectors of shape
        `[N, num_heads*irreps_head_dict[key]]` for each key in `irreps_head_dict`.
        """
        super().__init__()
        self.irreps_head_dict: Dict[str, Irreps] = irreps_head_dict

        self.vec2heads_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
            f"AttnHeads2Vec({k})": AttnHeads2Vec(irreps_head=v)
            for k, v in self.irreps_head_dict.items()
            }
        )


    def forward(self, x_dict: Dict[str, Tensor]):
        out_dict: Dict[str, Tensor] = {
            k: self.vec2heads_dict[f"AttnHeads2Vec({k})"](v)
            for k, v in x_dict.items()
        }
        return out_dict




class DropoutModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps], p: float):
        r"""`irreps_in_dict`'s key is used to construct dictionary. It's value is not necessary.
        """
        super().__init__()
        self.irreps_in_dict: Dict[str, Irreps] = irreps_in_dict
        self.probability: float= p

        self.dropout_dict: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {
            f"Dropout({k})": torch.nn.Dropout(p=self.probability)
            for k, _ in self.irreps_in_dict.items()
            }
        )


    def forward(self, x_dict: Dict[str, Tensor]):
        out_dict: Dict[str, Tensor] = {
            k: self.dropout_dict[f"Dropout({k})"](v)
            for k, v in x_dict.items()
        }
        return out_dict




class SoftmaxScatterNormModuleDict(torch.nn.Module):
    def __init__(self, irreps_in_dict: Dict[str, Irreps]=None):
        r"""Only used for Irrep("0e") scalar.
            Notice that Input `irreps_in_dict` just to check whether the irreps are correct.
        It is not necessary.
        """
        super().__init__()
        assert if_only_contain_irrep(target=irreps_in_dict, irrep=Irrep("0e"))
        self.irreps_in_dict = irreps_in_dict
        


    def forward(self, edge_fea_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor], num_nodes_dict: Dict[str, int]):
        r"""Notice that `num_nodes_dict` used for scatter operation in softmax function
        to avoid error node feature shape because of unconnected(isolated) atoms.
        """
        out_dict = {
            k: softmax(
                src=v, index=edge_index_dict[k][1], dim=0,
                num_nodes=num_nodes_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[1]]
                )
            for k, v in edge_fea_dict.items()
        } # key: edge type, value shape=(num_edges, num_heads, irreps.dim) Or (num_edges, irreps.dim)
        # two [1]: pick dst node index from [src, node]
        # edge_fea_dict, edge_index_dict key: edge type; num_nodes_dict key: node type
        return out_dict




class ElementLevelScatterModuleDict(torch.nn.Module):
    def __init__(self, irreps_message_dict: Dict[str, Irreps], irreps_node_dict: Dict[str, Irreps]):
        r"""Linear(to adjust different shape of each edge type) -> ElementLevelSatter
        
        Default Scatter Parameters In this Model: dim=0, reduce="mean", out=None.
        """
        super().__init__()
        self.irreps_message_dict: Dict[str, Irreps] = irreps_message_dict   # key: edge type
        self.irreps_node_dict: Dict[str, Irreps] = irreps_node_dict         # key: node type

        self.irreps_linear_out_dict: Dict[str, Irreps] = {                  # key: edge type
            k: self.irreps_node_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[-1]]
            for k, _ in self.irreps_message_dict.items()
        }
        self.linear_dict = LinearModuleDict(
            irreps_in_dict=irreps_message_dict,
            irreps_out_dict=self.irreps_linear_out_dict,
            bias=False
        )

    def forward(self, message_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor], num_nodes_dict: Dict[str, int]):
        r"""Linear(to adjust different shape of each edge type) -> ElementLevelSatter
        Default Scatter Parameters In this Model: dim=0, reduce="mean", out=None.

        TODO Forget source node of message in `out_dict`.

        `message_dict`: edge messages of each edge type
            key: edge type, "8-6"
            value shape: `(num_edges(current edge type src-dst), message_dim)`
        `edge_index_dict`: src-dst node index of the edges of each edge type
            key: edge type, "8-6"
            value shape: `(num_edges(current edge type src-dst), 2)`
        `num_nodes_dict`: total node number of each node type
            key: node type, "6"

        `out_dict`: scatter outcomes of each node type
            key: node type, "6"
            value shape: `(num_nodes, num_edge_types, irreps_node_dict[..].dim)`
        `num_type_dict`: non-zero vectors' number of each node type in `out_dict`, used as normalize number in `TypeLevelScatterModuleDict`.
            key: node type, "6"
            value shape: `(num_nodes, )`
            """
        message_dict = self.linear_dict(x_dict=message_dict)

        out_dict: Dict[str, List[Tensor]] = {
            k: []
            for k, _ in num_nodes_dict.items()
        }
        num_types_dict: Dict[str, List[Tensor]] = {
            k: []
            for k, v in num_nodes_dict.items()
        }
        
        for k, v in message_dict.items():
            out_scatter: Tensor = scatter(
                src=v, 
                index=edge_index_dict[k][1],  #[1]: pick dst index from [src, dst]
                dim=0, 
                dim_size=num_nodes_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[-1]], 
                reduce="mean"
            )
            detach_out_scatter: Tensor = out_scatter.detach()
            num_type_scatter: Tensor = \
                (detach_out_scatter!=torch.zeros_like(detach_out_scatter[0]))
            while(num_type_scatter.dim()>=2): # support num_heads dim
                num_type_scatter = num_type_scatter.any(dim=-1).float()
            num_type_scatter = out_scatter.new_tensor(num_type_scatter)
            out_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[-1]].append(out_scatter)
            num_types_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[-1]].append(num_type_scatter)
            
        """out_dict
        begin: "6": []
        append: "6": [
                (edge-type 1)Tensor[(node 1): Tensor, (node 2): Tensor, ...]
            ]
        append: "6": [
                (edge-type 1)Tensor[(node 1): Tensor,  (node 2): Tensor,  ...],
                (edge-type 2)Tensor[(node 1): Zeros,  (node 2): Tensor,  ...],
            ]
        ...
        shape: (num_edge_types("6"), num_nodes("6"), irreps_node_dict["6"].dim) Or
        shape: (num_edge_types("6"), num_nodes("6"), num_heads, irreps_node_dict["6"].dim)
        Notice that Tensor may equal to Zero Vector !

        num_types_dict
        "6": [
            (edge-type 1)Tensor[(node 1): 1.0,  (node 2): 1.0,  ...],
            (edge-type 2)Tensor[(node 1): 0.0,  (node 2): 1.0,  ...],
            ...
        ]
        shape: (num_edge_types("6"), num_nodes("6")) even if there is num_heads dim in out_dict
            
        """
        
        out_dict: Dict[str, Tensor] = {
            k: torch.stack(v, dim=1)
            for k, v in out_dict.items()
        }
        num_types_dict: Dict[str, Tensor] = {
            k: torch.sum(torch.stack(v, dim=1), dim=1)
            for k, v in num_types_dict.items()
        }
        """out_dict
        begin: 
        "6": [
                (edge-type 1)Tensor[(node 1): Tensor,  (node 2): Tensor,  ...],
                (edge-type 2)Tensor[(node 1): Zeros,  (node 2): Tensor,  ...],
            ]

        stack: 
        "6": Tensor[
                (node 1): Tensor[(edge-type 1)Tensor, (edge-type 2)Zeros,...], 
                (node 2): Tensor[(edge-type 1)Tensor, (edge-type 2)Tensor,...],
                ...
            ]
        shape: (num_nodes("6"), num_edge_types("6"), irreps_node_dict["6"].dim) Or
        shape: (num_nodes("6"), num_edge_types("6"), num_heads, irreps_node_dict["6"].dim)
        Notice that Tensor may equal to Zero Vector !

        num_types_dict
        stack: 
        "6": Tensor[
                (node 1): Tensor[(edge-type 1)1.0, (edge-type 2)0.0,...], 
                (node 2): Tensor[(edge-type 1)1.0, (edge-type 2)1.0,...],
                ...
            ]
        sum:
        "6": Tensor[(node 1): 1.0,  (node 2): 2.0,  ...]
        shape: (num_nodes("6"),)
        """

        return out_dict, num_types_dict




class TypeLevelScatterModuleDict(torch.nn.Module):
    def __init__(self):
        r"""TypeLevelScatter and get node features. Notice that a Linear is needed when irreps 
        of type message is not equal to irreps of node features.
        """
        super().__init__()
        self.eps: float = 1e-5 # used when normalize to avoid divide-by-zero error.

    def forward(self, type_message_dict: Dict[str, Tensor], num_types_dict: Dict[str, Tensor]):
        r"""
        `type_message_dict`: key: node type, "6"
            value shape: `(num_nodes, num_edge_types, irreps_type_message_dict[..].dim)` or 
        `(num_nodes, num_edge_types, num_heads, irreps_type_message_dict[..].dim)`
        `num_type_dict`: non-zero vectors' number of each node type in `out_dict`, 
        used as normalize number in `TypeLevelScatterModuleDict`.
            key: node type, "6"
            value shape: `(num_nodes, )`
            """
        
        out_dict: Dict[str, Tensor] = {
            k: torch.sum(v, dim=1)/(num_types_dict[k]+self.eps).reshape(-1, *[1 for _ in range(v.dim()-2)]) # =unsqueeze several times until the shape is correct
            for k, v in type_message_dict.items()
        }

        return out_dict




class TransformerModuleDict(torch.nn.Module):
    def __init__(self, 
                 irreps_node_fea_dict: Dict[str, Irreps], 
                 irreps_edge_fea_dict: Dict[str, Irreps], 
                 irreps_edge_vec_embed: Irreps, 
                 irreps_edge_length_embed: Irreps,
                 num_heads: int, 
                 alpha_dropout: float
                 ):
        r"""
            Notice that all values in `irreps_edge_fea_dict` should be divisible by `num_heads`.
        For example, Irreps("16x0e+8x1e+4x2e") // 4 = Irreps("4x0e+2x1e+1x2e").

            Notice that each edge type has the same `irreps_edge_vec_embed` and 
            `irreps_edge_length_embed`.
        
        """
        super().__init__()
        self.irreps_node_fea_dict: Dict[str, Irreps] = irreps_node_fea_dict
        self.irreps_edge_fea_dict: Dict[str, Irreps] = irreps_edge_fea_dict
        self.irreps_edge_vec_embed: Irreps = irreps_edge_vec_embed
        self.irreps_edge_length_embed: Irreps = irreps_edge_length_embed 
        self.num_heads: int = num_heads
        self.alpha_dropout: float = alpha_dropout

        self.irreps_message_head_dict: Dict[str, Irreps] = {
            k: divide_irreps(irreps=v, num_div=self.num_heads)
            for k, v in self.irreps_edge_fea_dict.items()
        }
        self.irreps_alpha_head_dict: Dict[str, Irreps] = {
            k: to_dtype_0e_scalar_irreps(get_mul_0(v), dtype=Irreps) 
            for k, v in self.irreps_message_head_dict.items()
            } # only Irrep="0e" in alpha irreps
        # = 1 =
        self.node_src_linear_dict: torch.nn.Module = LinearModuleDict(
            irreps_in_dict={
                m: v
                for m, _ in self.irreps_edge_fea_dict.items()
                for k, v in self.irreps_node_fea_dict.items()
                if k==to_dtype_nodes_from_edge(edge=m, dtype=[Tuple, str])[0]
            },  # key from node(src) type to edge type #[0]: pick out src node index from [src, dst]
            irreps_out_dict=self.irreps_edge_fea_dict, 
            bias=True
        )
        self.node_dst_linear_dict: torch.nn.Module = LinearModuleDict(
            irreps_in_dict={
                m: v
                for m, _ in self.irreps_edge_fea_dict.items()
                for k, v in self.irreps_node_fea_dict.items()
                if k==to_dtype_nodes_from_edge(edge=m, dtype=[Tuple, str])[1]
            },  # key from node(dst) type to edge type #[1]: pick out dst node index from [src, dst]
            irreps_out_dict=self.irreps_edge_fea_dict, 
            bias=False
        )
        self.message_dtp_dict: torch.nn.Module = DepthwiseTensorProductModuleDict(
            irreps_in_dict=self.irreps_edge_fea_dict, 
            irreps_edge_vec_embed=self.irreps_edge_vec_embed,
            dtp_internal_weights=False,
            irreps_edge_length_embed=self.irreps_edge_length_embed,
            mlp_irreps_mid_list=[Irreps("64x0e"), Irreps("64x0e")],
            mlp_add_last_linear="bias",
            mlp_if_act=True,
            mlp_if_norm=True,
            mlp_act_scalars=torch.nn.SiLU(),
            mlp_act_gates=torch.nn.Sigmoid(),
            mlp_norm_type="layer"
        )
        self.message_vec2heads_dict: torch.nn.Module = Vec2AttnHeadsModuleDict(
            irreps_head_dict=self.irreps_message_head_dict, num_head=self.num_heads
        )
        # = 2 =
        self.head_message_mlp_to_alpha_dict: torch.nn.Module = MultiLayerPerceptionModuleDict(
            irreps_in_dict=self.irreps_message_head_dict,
            irreps_out_dict=self.irreps_alpha_head_dict,
            irreps_mid_list=[],
            add_last_linear=None,
            if_act=True, 
            if_norm=True,
            act_scalars=torch.nn.SiLU(),
            act_gates=None,
            norm_type="layer"
        )
        self.head_alpha_linear: torch.nn.Module= LinearModuleDict(
            irreps_in_dict=self.irreps_alpha_head_dict, 
            irreps_out_dict={k: Irreps("1x0e") for k, _ in self.irreps_alpha_head_dict.items()}, 
            bias=False
        )
        self.head_alpha_softmaxnorm: torch.nn.Module = SoftmaxScatterNormModuleDict(
            irreps_in_dict=self.irreps_alpha_head_dict
        )

        self.head_alpha_dropout: torch.nn.Module = DropoutModuleDict(
            irreps_in_dict=self.irreps_alpha_head_dict, p=self.alpha_dropout
        )
        # = 3 =
        self.head_message_mlp_to_value_dict: torch.nn.Module = MultiLayerPerceptionModuleDict(
            irreps_in_dict=self.irreps_message_head_dict,
            irreps_out_dict=self.irreps_message_head_dict,
            irreps_mid_list=[],
            add_last_linear=None,
            if_act=True, 
            if_norm=True,
            act_scalars=torch.nn.SiLU(),
            act_gates=torch.nn.Sigmoid(),
            norm_type="layer"
        )
        # = 4 =
        self.attn_heads2vec_dict: torch.nn.Module = AttnHeads2VecModuleDict(
            irreps_head_dict=self.irreps_message_head_dict)
        self.attn_element_level_scatter_dict: torch.nn.Module = ElementLevelScatterModuleDict(
            irreps_message_dict=self.irreps_edge_fea_dict,
            irreps_node_dict=self.irreps_node_fea_dict
        )
        self.attn_type_level_scatter_dict: torch.nn.Module = TypeLevelScatterModuleDict()
        """ 
        # edge features' shape has been adjusted to irreps of node features
        #   in self.attn_element_level_scatter_dict
        self.new_node_fea_linear_dict: torch.nn.Module = LinearModuleDict(
            irreps_in_dict=self.irreps_alpha_head_dict, 
            irreps_out_dict={k: Irreps("1x0e") for k, _ in self.irreps_alpha_head_dict}, 
            bias=False
        )"""
        
        


    def forward(self, edge_fea_dict: Dict[str, Tensor], 
                node_fea_dict: Dict[str, Tensor], 
                edge_index_dict: Dict[str, Tensor], 
                edge_vec_embed_dict: Dict[str, Tensor], 
                edge_length_embed_dict: Dict[str, Tensor],
                num_nodes_dict: Dict[str, int]):
        r"""
            1. `node_fea(src), node_fea(dst) --Linear--> node_fea(src), node_fea(dst) 
        --Add-edge_fea--> message --DTP--> message --Vec2Heads--> message(head)`;
            2. `message(head)--MLP(for"0e")--> alpha --Linear(DotProduct)--> alpha 
        --SoftmaxNorm--> alpha --Dropout--> alpha`;
            3. `message(head)--MLP--> value`;
            4. `alpha, value --Mul--> attn(head) --Attnheads2Vec--> attn(Output!) 
        --((Linear-)Element+Type)LevelScatter--> node_fea(dst)(Output!)`.

            Notice that two output `attn(Output!)` is new `edge_fea`, 
        `node_fea(dst)(Output!)` is new `node_fea`.

            Notice that num_nodes_dict used for scatter operation in softmax and scatter function
        to avoid error node feature shape because of unconnected(isolated) atoms.
            """
        # = 1 =
        node_src_fea_dict: Dict[str, Tensor] = {
            k: node_fea_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[0]][v[0]] #two [0]: pick src from [src, dst] in edge type and edge index
            for k, v in edge_index_dict.items()
        } # key: edge type
        node_dst_fea_dict: Dict[str, Tensor] = {
            k: node_fea_dict[to_dtype_nodes_from_edge(edge=k, dtype=[Tuple, str])[1]][v[1]] #two [1]: pick dst node index from [src, dst] in edge type and edge index
            for k, v in edge_index_dict.items()
        } # key: edge type
        node_src_fea_dict = self.node_src_linear_dict(x_dict=node_src_fea_dict)
        node_dst_fea_dict = self.node_dst_linear_dict(x_dict=node_dst_fea_dict)
        message_dict: Dict[str, Tensor] = {
            k: v + node_src_fea_dict[k] + node_dst_fea_dict[k]
            for k, v in edge_fea_dict.items()
        }
        message_dict = self.message_dtp_dict(
            edge_fea_dict=message_dict, 
            edge_vec_dict=edge_vec_embed_dict, 
            edge_length_dict=edge_length_embed_dict)
        message_head_dict = self.message_vec2heads_dict(x_dict=message_dict)

        # = 2 =
        alpha_dict: Dict[str, Tensor] = self.head_message_mlp_to_alpha_dict(x_dict=message_head_dict)
        alpha_dict = self.head_alpha_linear(alpha_dict)
        # key: edge type, value shape=(num_edges, num_heads, irreps.dim) --> (num_edges, num_heads, 1)
        alpha_dict = self.head_alpha_softmaxnorm(
            edge_fea_dict=alpha_dict, 
            edge_index_dict=edge_index_dict, 
            num_nodes_dict=num_nodes_dict)
        # key: edge type, value shape=(num_edges, num_heads, 1)
        # two [1]: pick dst node index from [src, node]
        # alpha_dict, edge_index_dict key: edge type
        # num_nodes_dict key: node type
        alpha_dict = self.head_alpha_dropout(x_dict=alpha_dict)

        # = 3 =
        value_dict: Dict[str, Tensor] = self.head_message_mlp_to_value_dict(x_dict=message_head_dict)

        # = 4 =
        attn_dict: Dict[str, Tensor] = {
            k: alpha * value_dict[k]
            for k, alpha in alpha_dict.items()
        } # key: edge type, value shape=(num_edges, num_heads, irreps_message_head.dim)
        attn_dict = self.attn_heads2vec_dict(x_dict=attn_dict) # Output !
        
        mid_dict, num_types_dict = self.attn_element_level_scatter_dict(
            message_dict=attn_dict, # no problem without clone ?
            edge_index_dict=edge_index_dict, 
            num_nodes_dict=num_nodes_dict
            )
        new_node_fea_dict: Dict[str, Tensor] = self.attn_type_level_scatter_dict(
            type_message_dict=mid_dict, 
            num_types_dict=num_types_dict
        )

        return attn_dict, new_node_fea_dict # new-edge-fea, new-node-fea






