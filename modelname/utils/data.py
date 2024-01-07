from typing import List, Tuple, Type, Any, Dict
from e3nn.o3 import Irreps, Irrep


# Data Type Transfor Tools For pair_type (or edge) in Block Dict
def to_dtype_edge(edge: str | Tuple[int, int] | Tuple[str, str, str], 
            dtype: Type | List[Type]) -> str | Tuple[int, int] | Tuple[str, str, str]:
    """Transform Data Type of edge (or pair type) in edge feature's, edge index's and y's Dict.
    
    dtype: 
        `str`: `str`
        `[Tuple, int]`: `Tuple[int, int]`
        `[Tuple, str]`: `Tuple[str, str, str]`

    The Data Type of edge identity varies (with example):
        HeteroDataset (graph.py): `Tuple[str, str, str]`, `('83', '83-83', '83')`\n
        Model (model.py): `str`, `'83-83'` (torch.nn.ModuleDict should not contain Tuple as key, only str)\n
        blocks_dict (graph.py): `Tuple[int, int]`, `(83, 83)`\n
        model.json (graph.py): `str`, `'83-83'`\n
    """
    if type(edge)==type("83-83"): # isinstance(edge, str):
        strs: List[str] = edge.split('-')
        p, q = strs
        p, q = int(p), int(q)
    elif type(edge)==type(tuple([None])): # isinstance(edge, Tuple): # Notice that edge can not change
        if type(edge[0])==type((83, 83)[0]):# isinstance(edge[0], int) and len(edge)==2:
            p, q = edge
        elif type(edge[0])==type(("83", "83-83", "83")[0]): # isinstance(edge[0], str) and len(edge)==3:
            p, _, q = edge
            p, q = int(p), int(q)
        else:
            raise NotImplementedError(f"Unsupported Input Data Type of Edge (or Pair Type) Element.\n"
                                  f"Notice that the Input Data Type of Edge Element: type(edge[0])={type(edge[0])}.")
    else:
        raise NotImplementedError(f"Unsupported Input Data Type of Edge (or Pair Type).\n"
                                  f"Notice that the Input Data Type of Edge: type(edge)={type(edge)}.")
    
    if dtype == str:
        return str(p)+'-'+str(q)
    elif dtype == [Tuple, int]:
        return (p, q)
    elif dtype == [Tuple, str]:
        p, q = str(p), str(q)
        return (p, p+'-'+q, q)
    else:
        raise NotImplementedError(f"Unsupported Output Data Type of Edge (or Pair Type).\n"
                                  f"Notice that the Output Data Type of Edge (or Pair Type): dtype={dtype}")


def to_dtype_node(node: str | int, 
            dtype: Type) -> str | int:
    """Transform Data Type of node in node feature's Dict.
    
    dtype: 
        `str`: `str`
        `int`: `int`

    The Data Type of edge identity varies (with example):
        HeteroDataset (graph.py): `int`, `83`\n
        Model (model.py): `str`, `"83"` (torch.nn.ModuleDict should not contain Tuple as key, only str)\n
        blocks_dict (graph.py): `int`, `83`\n
    """
    if type(node)==type("83"): # isinstance(node, str):
        node: int = int(node)
    elif type(node)==type(83): # isinstance(node, int): 
        node: int = node
    else:
        raise NotImplementedError(f"Unsupported Input Data Type of Node.\n"
                                  f"Notice that the Input Data Type of Node: type(node)={type(node)}.")
    
    if dtype == str:
        return str(node)
    elif dtype == int:
        return node
    else:
        raise NotImplementedError(f"Unsupported Output Data Type of Node.\n"
                                  f"Notice that the Output Data Type of Node: dtype={dtype}")


def to_dtype_nodes_from_edge(edge: str | Tuple[int, int] | Tuple[str, str, str],
                       dtype: List[Type]) -> Tuple[int] | Tuple[str]:
    """Get src,dst nodes from Data Type of edge (or pair type) in Block Dict.
    
    dtype: 
        `[Tuple, int]`: `Tuple[int, int]`
        `[Tuple, str]`: `Tuple[str, str]`

    Return `(src_node_id, dst_node_id)`.
    """
    edge: Tuple[int, int] = to_dtype_edge(edge, dtype=[Tuple, int])
    p, q = edge
    
    if dtype == [Tuple, int]:
        return (p, q)
    elif dtype == [Tuple, str]:
        p, q = str(p), str(q)
        return (p, q)
    else:
        raise NotImplementedError(f"Unsupported Output Data Type of src,dst Nodes.\n"
                                  f"Notice that the Output Data Type of src,dst Nodes: dtype={dtype}")


def to_dtype_0e_scalar_irreps(irreps: int | str | Irreps, dtype=Type) -> int | str | Irreps:
    """Transform Data Type of Irrep="0e" scalar vector's irreps.
    
    dtype: 
        `int`: `int`, `5`
        `str`: `str`, `"4x0e+1x0e"`
        `Irreps`: `Irreps`, `Irreps("4x0e+1x0e")`

    For `Irrep="0e"` scalar vectors, \n
    Irreps=`Irreps("Ax0e+Bx0e")` can be simply record as one integer (A+B).

    Notice that non-simplified Irreps or string will be transformed into simplified form.
    When apply transform: non-simplified Irreps <--> string, do not use this function.
    """
    if type(irreps)==type(4): # isinstance(edge, int):
        irreps: int = irreps
    elif type(irreps)==type("4x0e+1x0e"): # isinstance(edge, str): 
        irreps: int = int(
            str(Irreps(irreps).simplify())[:-3] #[:-3]: pick "{}"" from "{}x0e"
            ) 
    elif type(irreps)==type(Irreps("4x0e+1x0e")): # isinstance(edge, Irreps): 
        irreps: int = int(
            str(irreps.simplify())[:-3] #[:-3]: pick "{}"" from "{}x0e"
        )
    else:
        raise NotImplementedError(f"Unsupported Input Data Type of Irrep=\"0e\" scalar vector.\n"
                                  f"Notice that the Input Data Type of irreps: type(irreps)={type(irreps)}.")
    if dtype == int:
        return irreps
    elif dtype == str: # simplified form
        return str(irreps) + "x0e"
    elif dtype == Irreps: # simplified form
        return Irreps(str(irreps) + "x0e")
    else:
        raise NotImplementedError(f"Unsupported Output Data Type of Irrep=\"0e\" scalar vector.\n"
                                  f"Notice that the Output Data Type of Irrep=\"0e\" scalar vector: dtype={dtype}")


def divide_irreps(irreps: Irreps, num_div: int) -> Irreps:
    div_irreps: Irreps = Irreps("")
    for mul, ir in irreps:
        mul: float = mul / num_div
        if mul.is_integer()==False:
            raise ValueError(f"Input Irreps={irreps} Is Not Divisible By num_div={num_div}.")
        mul: int = int(mul)
        strs: str = str(mul) + "x" + str(ir)
        div_irreps = div_irreps + Irreps(strs)
    return div_irreps


def if_all_include_irrep(target: Irreps | Dict[Any, Irreps] | List[Irreps], irrep: Irrep) -> bool:
    r"""Judge whether `irrep` is included in the `target(Irreps)` or 
    judge whether `irrep` is included in every value in the `target(Dict[Any, Irreps])`.
    
    For example: `irrep=Irrep("0e")`
        `target=Irreps("1x0e+1x1e")`, return `True`;
        `target={"6-8": Irreps("1x0e+1x1e"), "8-6": Irreps("1x1e")}`, return `False`;
        `target={"6-8": Irreps("1x0e+1x1e"), "8-6": Irreps("1x0e")}`, return `True`;
    """
    if type(target)==type(Irreps("1x0e")): # isinstance(node, Irreps):
        target: List[Irreps] = [target]
    elif type(target)==type({"1": Irreps("1x0e")}): # isinstance(node, Dict): 
        target: List[Irreps] = [v for v in target.values()]
    elif type(target)==type([Irreps("1x0e")]): # isinstance(node, List): 
        target: List[Irreps] = target
    else:
        raise NotImplementedError(f"Unsupported Input Data Type of Target.\n"
                                  f"Notice that the Input Data Type of Target: type(target)={type(target)}.")
    
    is_true: bool = all(
        [(irrep in x) for x in target]
        )

    return is_true


def if_only_contain_irrep(target: Irreps | Dict[Any, Irreps] | List[Irreps], irrep: Irrep) -> bool:
    r"""Judge whether `irrep` is the only Irrep included in the `target(Irreps)` or 
    judge whether `irrep` is the only Irrep included in every value in the `target(Dict[Any, Irreps])`.
    
    For example: `irrep=Irrep("0e")`
        `target=Irreps("1x0e+1x1e")`, return `False`;
        `target={"6-8": Irreps("1x0e"), "8-6": Irreps("1x0e")}`, return `True`;
        `target={"6-8": Irreps("1x0e+1x1e"), "8-6": Irreps("1x0e")}`, return `False`;
    """
    if type(target)==type(Irreps("1x0e")): # isinstance(node, Irreps):
        target: List[Irreps] = [target]
    elif type(target)==type({"1": Irreps("1x0e")}): # isinstance(node, Dict): 
        target: List[Irreps] = [v for v in target.values()]
    elif type(target)==type([Irreps("1x0e")]): # isinstance(node, List): 
        target: List[Irreps] = target
    else:
        raise NotImplementedError(f"Unsupported Input Data Type of Target.\n"
                                  f"Notice that the Input Data Type of Target: type(target)={type(target)}.")
    """
    for k, v in xxdict.items():
        for mul, ir in v:
            assert ir == Irrep("0e")"""
    is_true: bool = all(
        [all([ir==irrep for mul, ir in x]) for x in target]
        )

    return is_true



