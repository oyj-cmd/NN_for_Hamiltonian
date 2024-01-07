from typing import Callable, List, Dict, Optional, Tuple, Set, Type
from collections import namedtuple
from torch import Tensor
from numpy.typing import NDArray
import os
import h5py
import json
import numpy as np
import torch
import logging
from e3nn.o3 import Irreps, wigner_3j
from torch_geometric.data import HeteroData


# = Initialize Logger =
logger = logging.getLogger('graph.py')


# = Tools = 
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


def irreps_from_l1l2(l1, l2, mul, spinful):
    r'''
    non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
    
    spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
    required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
    required_irreps = (1+2+3)x0 = 1+2+3
    required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    
    notice that required_irreps_x1 is a list of Irreps
    '''
    p = 1
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(mul, (l, p)) for l in required_ls])
    required_irreps_full = required_irreps
    required_irreps_x1 = None
    if spinful:
        required_irreps_x1 = []
        for _, ir in required_irreps:
            required_ls_irx1 = range(abs(ir.l - 1), ir.l + 1 + 1)
            irx1 = Irreps([(mul, (l, p)) for l in required_ls_irx1])
            required_irreps_x1.append(irx1)
            required_irreps_full += irx1
    return required_irreps_full, required_irreps, required_irreps_x1


def flt2cplx(flt_dtype):
    if flt_dtype == torch.float32:
        cplx_dtype = torch.complex64
    elif flt_dtype == torch.float64:
        cplx_dtype = torch.complex128
    elif flt_dtype == np.float32:
        cplx_dtype = np.complex64
    elif flt_dtype == np.float64:
        cplx_dtype = np.complex128
    else:
        raise NotImplementedError(f'Unsupported float dtype: {flt_dtype}')
    return cplx_dtype


class Rotate:
    def __init__(self, default_dtype_torch, device_torch='cpu', spinful=False):
        sqrt_2 = 1.4142135623730951
            
        self.spinful = spinful
        #TODO
        if spinful:
            assert default_dtype_torch in [torch.complex64, torch.complex128]
        else:
            assert default_dtype_torch in [torch.float32, torch.float64]
        
        # openmx的实球谐函数基组变复球谐函数
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat, device=device_torch),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]], dtype=torch.cfloat, device=device_torch),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch.cfloat, device=device_torch),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch.cfloat, device=device_torch),
        }
        # openmx的实球谐函数基组变wiki的实球谐函数 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch).to(device=device_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]].to(device=device_torch),
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]].to(device=device_torch),
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]].to(device=device_torch)
        }

    def wiki2openmx_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T


class e3TensorDecomp:
    def __init__(self, out_js_list, default_dtype_torch, spinful=False, device_torch='cpu'):
        if spinful:
            default_dtype_torch = flt2cplx(default_dtype_torch)
        self.spinful = spinful
        
        self.device = device_torch
        self.out_js_list = out_js_list

        required_irreps_out = Irreps(None)
        in_slices = [0]
        wms = [] # wm = wigner_multiplier
        H_slices = [0]
        wms_H = []
        if spinful:
            in_slices_sp = []
            H_slices_sp = []
            wms_sp = []
            wms_sp_H = []
        
        # 所有种类哈密顿不可约矩阵块的角动量对(H_l1, H_l2)被储存在out_js_list中
        for H_l1, H_l2 in out_js_list:
            
            # = construct required_irreps_out =
            mul = 1
            # 不可约表示的张量积H_l1 x H_l2变直和e3nn-Irreps形式：required_irreps_out_single, required_irreps_x1 
            _, required_irreps_out_single, required_irreps_x1 = irreps_from_l1l2(H_l1, H_l2, mul, spinful)
            required_irreps_out += required_irreps_out_single
            
            # spinful case, example: (1x0.5)x(2x0.5) = (1+2+3)x(0+1) = (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
            # everything on r.h.s. as a whole constitutes a slice in in_slices
            # each bracket on r.h.s. above corresponds to a slice in in_slice_sp
            # 考虑自旋的情况下，要进行2步直和变张量积（get_H）的操作，需要存储2套CG系数矩阵
            # 此处生成了第一套CG系数矩阵: wm_sp
            # 模型输出哈密顿向量: out_js_list所有种类不可约哈密顿矩阵的总和
            # 输出哈密顿向量中对其中一个不可约哈密顿矩阵切片的形状：required_irreps_out_single.dim + 
            #    required_irreps_x1[0].dim + required_irreps_x1[1].dim + required_irreps_x1[2].dim
            # 希望经过第一次操作后，形成自旋哈密顿矩阵的四个分块
            # 转化成形状:  required_irreps_out_single.dim x 4
            # ir.l x (0+1) = ir.l x 0 + ir.l x 1
            # ir.l x 1 <--wm_irx1-- ir_1.l(in ir_times_1)
            # ir.l x (0+1) (in required_irreps_out_single) <--
            #         wm_sp=[None, 3个wm_irx1]-- ir_1.l(in required_irreps_out_single, required_irreps_x1) 
            
            # in_slice_sp: required_irreps_out_single.dim, required_irreps_x13个元素.dim的累加
            if spinful:
                in_slice_sp = [0, required_irreps_out_single.dim]
                H_slice_sp = [0]
                wm_sp = [None]
                wm_sp_H = []
                for (_a, ir), ir_times_1 in zip(required_irreps_out_single, required_irreps_x1):
                    required_irreps_out += ir_times_1
                    in_slice_sp.append(in_slice_sp[-1] + ir_times_1.dim)
                    H_slice_sp.append(H_slice_sp[-1] + ir.dim)
                    wm_irx1 = []
                    wm_irx1_H = []
                    for _b, ir_1 in ir_times_1:
                        for _c in range(mul):
                            wm_irx1.append(wigner_3j(ir.l, 1, ir_1.l, dtype=default_dtype_torch, device=device_torch))
                            wm_irx1_H.append(wigner_3j(ir_1.l, ir.l, 1, dtype=default_dtype_torch, device=device_torch) * (2 * ir_1.l + 1))
                            # wm_irx1.append(wigner_3j(ir.l, 1, ir_1.l, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir_1.l + 1))
                            # wm_irx1_H.append(wigner_3j(ir_1.l, ir.l, 1, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir_1.l + 1))
                    wm_irx1 = torch.cat(wm_irx1, dim=-1)
                    wm_sp.append(wm_irx1)
                    wm_irx1_H = torch.cat(wm_irx1_H, dim=0)
                    wm_sp_H.append(wm_irx1_H)
            
            # = construct slices =
            # in_slices，H_slices储存了每一种哈密顿不可约矩阵块的矩阵元总个数 的累加数
            # 两者是一样的，不知道为什么设置了2种变量
            # in_slices_sp，H_slices_sp则用于考虑自旋的情况
            in_slices.append(required_irreps_out.dim)
            H_slices.append(H_slices[-1] + (2 * H_l1 + 1) * (2 * H_l2 + 1))
            if spinful:
                in_slices_sp.append(in_slice_sp)
                H_slices_sp.append(H_slice_sp)
            
            # = get CG coefficients multiplier to act on net_out =
            # 考虑自旋的情况下，要进行2步直和变张量积的操作，需要存储2套CG系数矩阵
            # 此处生成了第二套CG系数矩阵: wm
            # 不考虑自旋的情况下，只需要1步操作，存储1套CG系数矩阵
            # 此处生成了CG系数矩阵: wm
            # 经过第一次操作的哈密顿对其中一个不可约哈密顿矩阵切片的形状: 
            # required_irreps_out_single.dim (x 4)
            # 此时不同自旋对应的矩阵块的操作相互独立
            # 希望一维变二维转化成形状: (2 x H_l1 + 1) x (2 x H_l2 + 1) x 4
            wm = []
            wm_H = []
            # 对当前哈密顿不可约矩阵块的直和e3nn-Irreps形式的不可约表示循环
            for _a, ir in required_irreps_out_single:
                for _b in range(mul):
                    # about this 2l+1: 
                    # we want the exact inverse of the w_3j symbol, i.e. torch.einsum("ijk,jkl->il",w_3j(l,l1,l2),w_3j(l1,l2,l))==torch.eye(...). but this is not the case, since the CG coefficients are unitary and w_3j differ from CG coefficients by a constant factor. but we know from https://en.wikipedia.org/wiki/3-j_symbol#Mathematical_relation_to_Clebsch%E2%80%93Gordan_coefficients that 2l+1 is exactly the factor we want.
                    wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch, device=device_torch))
                    wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch, device=device_torch) * (2 * ir.l + 1))
                    # wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir.l + 1))
                    # wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir.l + 1))
            wm = torch.cat(wm, dim=-1)
            wm_H = torch.cat(wm_H, dim=0)
            wms.append(wm)
            wms_H.append(wm_H)
            if spinful:
                wms_sp.append(wm_sp)
                wms_sp_H.append(wm_sp_H)
            
        # = check net irreps out =
        if spinful:
            required_irreps_out = required_irreps_out + required_irreps_out
        
        self.in_slices = in_slices
        self.wms = wms
        self.H_slices = H_slices
        self.wms_H = wms_H
        if spinful:
            self.in_slices_sp = in_slices_sp
            self.H_slices_sp = H_slices_sp
            self.wms_sp = wms_sp
            self.wms_sp_H = wms_sp_H

        # = register rotate kernel =
        self.rotate_kernel = Rotate(default_dtype_torch, spinful=spinful, device_torch=device_torch)
        
        if spinful:
            sqrt2 = 1.4142135623730951
                                            #  0,   y,   z,   x
            # self.oyzx2spin = torch.tensor([[   0, -1, 1j,   0,  ],  # uu
            #                                [   1,  0,  0,   1,  ],  # ud
            #                                [  -1,  0,  0,   1,  ],  # du
            #                                [   0,  1, 1j,   0,  ]], # dd
            #                                dtype=default_dtype_torch) / sqrt2
            self.oyzx2spin = torch.tensor([[  1,   0,   1,   0],
                                           [  0, -1j,   0,   1],
                                           [  0,  1j,   0,   1],
                                           [  1,   0,  -1,   0]],
                                            dtype=default_dtype_torch, device=device_torch) / sqrt2
        
    
    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        if self.spinful:
            half_len = int(net_out.shape[-1] / 2)
            re = net_out[:, :half_len]
            im = net_out[:, half_len:]
            net_out = re + 1j * im
        out = []
        for i in range(len(self.out_js_list)):
            # 考虑自旋的in_slice的切片长度是不考虑自旋的in_slice的4倍
            in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            net_out_block = net_out[:, in_slice]
            if self.spinful:
                # (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4) -> (1+2+3)x(0+1)
                H_block = []
                # self.wms_sp[i]是某个不可约矩阵块对应的第一套CG系数矩阵wm_sp=[None, 3个wm_irx1]
                for j in range(len(self.wms_sp[i])):
                    in_slice_sp = slice(self.in_slices_sp[i][j], self.in_slices_sp[i][j + 1])
                    if j == 0:
                        H_block.append(net_out_block[:, in_slice_sp].unsqueeze(-1))
                    else:
                        H_block.append(torch.einsum('jkl,il->ijk', self.wms_sp[i][j], net_out_block[:, in_slice_sp]))
                # cat以后H_block的形状：(ijk) num_blocks x required_irreps_out_single.dim x 4
                H_block = torch.cat([H_block[0], torch.cat(H_block[1:], dim=-2)], dim=-1)
                # (1+2+3)x(0+1) -> (uu,ud,du,dd)x(1x2)
                # 各自维度的长度记录
                # i: num_blocks, m: required_irreps_out_single.dim, n: 4
                # k: out_js_list中的角动量2*H_l1+1的求和, l: out_js_list中的角动量2*H_l2+1的求和
                # 注意out_js_list中的角动量H_l2个数=out_js_list中的角动量H_l1个数=out_js_list中的角动量对个数
                # j: 4, n: 4
                # eisum以后H_block的形状:(ijkl) num_blocks x 4 x out_js_list中的角动量2*H_l1+1的求和 x out_js_list中的角动量2*H_l2+1的求和
                # H_block很可能有极大的冗余，因为一个原子对分块哈密顿矩阵的角动量对个数很可能远低于out_js_list中的角动量对个数，这是同质图的弊端
                H_block = torch.einsum('imn,klm,jn->ijkl', H_block, self.wms[i], self.oyzx2spin)
                H_block = self.rotate_kernel.wiki2openmx_H(H_block, *self.out_js_list[i])
                out.append(H_block.reshape(net_out.shape[0], 4, -1))
            else:
                H_block = torch.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1)
                H_block = self.rotate_kernel.wiki2openmx_H(H_block, *self.out_js_list[i])
                out.append(H_block.reshape(net_out.shape[0], -1))
        return torch.cat(out, dim=-1) # output shape: [edge, (4 spin components,) H_flattened_concatenated]

    def get_net_out(self, H):
        r'''get net output from openmx type H'''
        out = []
        for i in range(len(self.out_js_list)):
            H_slice = slice(self.H_slices[i], self.H_slices[i + 1])
            l1, l2 = self.out_js_list[i]
            if self.spinful:
                H_block = H[..., H_slice].reshape(-1, 4, 2 * l1 + 1, 2 * l2 + 1)
                H_block = self.rotate_kernel.openmx2wiki_H(H_block, *self.out_js_list[i])
                # (uu,ud,du,dd)x(1x2) -> (1+2+3)x(0+1)
                H_block = torch.einsum('ilmn,jmn,kl->ijk', H_block, self.wms_H[i], self.oyzx2spin.T.conj())
                # (1+2+3)x(0+1) -> (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
                net_out_block = [H_block[:, :, 0]]
                for j in range(len(self.wms_sp_H[i])):
                    H_slice_sp = slice(self.H_slices_sp[i][j], self.H_slices_sp[i][j + 1])
                    net_out_block.append(torch.einsum('jlm,ilm->ij', self.wms_sp_H[i][j], H_block[:, H_slice_sp, 1:]))
                net_out_block = torch.cat(net_out_block, dim=-1)
                out.append(net_out_block)
            else:
                H_block = H[:, H_slice].reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
                H_block = self.rotate_kernel.openmx2wiki_H(H_block, *self.out_js_list[i])
                net_out_block = torch.sum(self.wms_H[i][None, :, :, :] * H_block[:, None, :, :], dim=(-1, -2))
                out.append(net_out_block)
        out = torch.cat(out, dim=-1)
        if self.spinful:
            out = torch.cat([out.real, out.imag], dim=-1)
        return out


# = My codes start here = 
# = Tools = 
def old_0_revert_blocks_dict(out_dict: Dict[Tuple[int, int], Tensor], 
        Ls_dict: Dict[int, List[int]], 
        is_spin: bool,
        default_dtype_torch: torch.dtype
        ) -> Dict[Tuple[int, int], Tensor]:
    """Test function.
    
    Revert from output of GNN model 
    `(hamiltonian block in direct sum)`
    to blocks_dict 
    `(origin hamiltonian block in tensor product)`.

    if is_spin=True, 
    Tensor shape: `(num_blocks, 2, 4, p_num_basis, q_num_basis)`
    Note that it is in 2 parts: real, imag, and the element is not complex number.
    Note that the real part comes first in dimension 0, then comes the imag part.

    if is_spin=False,
    Tensor shape: `(num_blocks, p_num_basis, q_num_basis)`
    """
    blocks_dict: Dict[Tuple[int, int], Tensor] = {}

    for pair_type, irreps in out_dict.items():
        # ================0.得到矩阵切片================
        # 从Ls_dict得到（当前原子对类型下）原子基函数角动量信息
        # 我们认为相同元素类型的原子基函数角动量信息完全相同, 所以从第一个atom_pair得到Ls
        p, q = pair_type
        p_Ls: NDArray[np.int] = np.array(Ls_dict[p])
        q_Ls: NDArray[np.int] = np.array(Ls_dict[q])
        p_num_basis_group: int = len(p_Ls)
        q_num_basis_group: int = len(q_Ls)
        p_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * p_Ls + 1), obj=0, values=0) # 往第一个位置插入0
        q_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * q_Ls + 1), obj=0, values=0)
        L_tuples: List[Tuple[int, int]] = []
        tuple_slices: List[int] = [0]
        # 从p_Ls, q_Ls得到上一个阶段矩阵irreps中按照特定顺序排布的一维基函数组对分块矩阵的irreps中各个基函数组的角动量对L_tuples
        # 先p后q，顺序和下面一样
        for p_single_L in p_Ls:
            for q_single_L in q_Ls:
                L_tuples.append(tuple([int(p_single_L), int(q_single_L)])) # 若不int()就是np.int64, 报错, out_js_list中的整数必须是python的int 
                num_basis_L_tuple: int = (1 + 2 * int(p_single_L)) * (1 + 2 * int(q_single_L))
                tuple_slices.append(tuple_slices[-1] + num_basis_L_tuple)
        """ 
        # ================2.重排矩阵元顺序，按照原重叠矩阵Irreps复原================
        sort_idx: List[int] = sort_dict[pair_type]
        # if is_spin=True:
        #(before e3TensorDecomp) irreps (real): num_blocks x (2 x 4 x p_num_basis x q_num_basis)
        # if is_spin=False:
        # irreps (real): num_blocks x (p_num_basis x q_num_basis)
        # Note that len(sort_idx) == (2 x 4 x p_num_basis x q_num_basis) or (p_num_basis x q_num_basis)
        irreps: Tensor = irreps[:, sort_idx]
        """

        # ================2.笛卡尔坐标->球谐坐标的坐标变换================
        transformer = e3TensorDecomp(out_js_list=L_tuples, 
                                        default_dtype_torch=default_dtype_torch, 
                                        spinful=is_spin) 
        irreps = transformer.get_H(irreps)
        # 只留下tuple_slices, p_num_basis_group, q_num_basis_group, p_slices, q_slices (和blocks_dict, irreps, pair_type)
        del p, q, p_Ls, q_Ls, p_single_L, q_single_L, 
        del transformer, num_basis_L_tuple, L_tuples

        # ================3.改变分块矩阵形状重新排序================
        # 从原子对分块矩阵blocks借助p_slices, q_slices，tuple_slices得到原始二维基函数组对分块矩阵的blocks
        num_blocks: int = irreps.shape[0] # number of blocks of the current atom pair type
        if is_spin==True:
            blocks: Tensor = torch.zeros((num_blocks, 4, p_slices[-1], q_slices[-1]), dtype=flt2cplx(default_dtype_torch)) 
        else:
            blocks: Tensor = torch.zeros((num_blocks, p_slices[-1], q_slices[-1]), dtype=default_dtype_torch)
        
        # 分别对原子p和q的不可约基函数组i和j循环，先p后q与先q后p可以导致完全不同顺序排布
        for i in range(p_num_basis_group): 
            i_slice = slice(p_slices[i], p_slices[i+1]) # p_slices的长度为（1+p_num_basis_group）
            i_len: int = i_slice.stop - i_slice.start
            for j in range(q_num_basis_group):
                j_slice = slice(q_slices[j], q_slices[j+1]) # q_slices的长度为（1+q_num_basis_group）
                j_len: int = j_slice.stop - j_slice.start
                k = j + i * q_num_basis_group
                k_slice = slice(tuple_slices[k], tuple_slices[k+1])

                # reshape规则不同的话也可以导致完全不同顺序排布
                if is_spin==True:
                    # irreps (complex): num_blocks x 4 x (p_num_basis x q_num_basis)
                    # current blocks (complex): num_blocks x 4 x p_num_basis x q_num_basis
                    blocks[:, :, i_slice, j_slice] = irreps[:, :, k_slice].reshape(num_blocks, 4, i_len, j_len)
                    # bug code for test
                    # blocks[:, :, j_slice, i_slice] = irreps[:, :, k_slice].reshape(num_blocks, 4, j_len, i_len) 
                else:
                    # irreps (real): num_blocks x (p_num_basis x q_num_basis)
                    # blocks (real): num_blocks x p_num_basis x q_num_basis
                    blocks[:, i_slice, j_slice] = irreps[:, k_slice].reshape(num_blocks, i_len, j_len)

        # 只留下blocks, (和blocks_dict, pair_type)
        del tuple_slices, p_num_basis_group, q_num_basis_group, p_slices, q_slices
        del irreps, num_blocks, i, j, i_slice, j_slice, k, k_slice, i_len, j_len


        # ================4.给blocks_dict赋值================
        if is_spin==True:
            # = break complex into real, imag parts = 
            #Before Tensor shape:  (num_blocks, 4, p_num_basis, q_num_basis)
            #After Tensor shape: (num_blocks, 2, 4, p_num_basis, q_num_basis)
            blocks_dict[pair_type] = torch.stack([blocks.real, blocks.imag], dim=1)
        else:
            # blocks (real): num_blocks x p_num_basis x q_num_basis
            blocks_dict[pair_type] = blocks
    
    return blocks_dict


# Data Type Transfor Tools For default_torch_dtype
def to_dtype_default_torch_dtype(default_torch_dtype: str | torch.dtype, 
    dtype: Type) -> str | torch.dtype:
    """Transform Data Type of default_torch_dtype.
    
    dtype: 
        `str`: `str`
        `torch.dtype`: `torch.dtype`

    Example: torch.float32 <-> 'torch.float32'
    """
    if type(default_torch_dtype)==type(torch.float32): # isinstance(default_torch_dtype, torch.dtype):
        i = default_torch_dtype
    elif type(default_torch_dtype)==type("torch.float32"): # isinstance(default_torch_dtype, str):
        if default_torch_dtype=="torch.float32":
            i = torch.float32
        elif default_torch_dtype=="torch.float64":
            i = torch.float64
        else:
            raise NotImplementedError(f"Unsupported default_torch_dtype: {default_torch_dtype}.")
    else:
        raise NotImplementedError(f"Unsupported Input Data Type of default_torch_dtype.\n"
           f"Notice that the Input Data Type of default_torch_dtype: type(default_torch_dtype)={type(default_torch_dtype)}")
    
    if dtype == str:
        return str(i)
    elif dtype == torch.dtype:
        return i
    else:
        raise NotImplementedError(f"Unsupported Output Data Type of default_torch_dtype.\n"
                                  f"Notice that the Output Data Type of default_torch_dtype: dtype={dtype}")


'''
def old_str_to_torch_dtype_default_torch_dtype(str_dtype: str) -> torch.dtype:
    if str_dtype=="torch.float32":
        torch_dtype = torch.float32
    elif str_dtype=="torch.float64":
        torch_dtype = torch.float64
    else:
        raise NotImplementedError
    return torch_dtype
'''


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


'''
def old_tuple_to_str_pair_type(tuple_pair_type: Tuple[int, int]) -> str:
    """Example: (83, 83) to '83-83'."""
    p, q = tuple_pair_type
    return f"{p}-{q}"


def old_str_to_tuple_pair_type(str_pair_type: str) -> Tuple[int, int]:
    """Example, '83-83' to (83, 83)."""
    strs: List[str] = str_pair_type.split('-')
    return (int(strs[0]), int(strs[1]))


def old_tuple_to_data_pair_type(tuple_pair_type: Tuple[int, int]) -> Tuple[str, str, str]:
    p, q = str(tuple_pair_type[0]), str(tuple_pair_type[1]) # p, q are str here
    data_pair_type: Tuple[str, str, str] = (p, p+"-"+q, q)
    return data_pair_type


def old_data_to_tuple_pair_type(data_pair_type: Tuple[str, str, str]) -> Tuple[int, int]:
    p, _, q = data_pair_type # p, q are str here
    tuple_pair_type: Tuple[int, int] = (int(p), int(q))
    return tuple_pair_type


def old_str_to_data_pair_type(str_pair_type: str) -> Tuple[str, str, str]:
    strs: List[str] = str_pair_type.split('-')
    return (strs[0], str_pair_type, str[1])


def old_data_to_str_pair_type(data_pair_type: Tuple[str, str, str]) -> str:
    return data_pair_type[1]
'''

# JSON File (graph.json) Helper in HeteroDataset
dataset_info_recorder = namedtuple(typename='dataset_info_recorder',
    field_names=[
        "is_spin", "default_dtype_torch", 
        "block_irreps_dict", 
        "simple_sorted_block_irreps_dict",
        "sort_dict",
        "inv_sort_dict",
        "Ls_dict"
    ])


def get_valid_pair_type(p_type: int, q_type: int) -> Tuple[int, int]:
        """Get VALID pair_type=(p_type, q_type) satisfying p_type <= q_type."""
        pair_type: Tuple[int, int] = None

        if p_type <= q_type:
            pair_type = (p_type, q_type)
        else:
            pair_type = (q_type, p_type)

        return pair_type


def is_edge_index_dict_equal(folder: str) -> bool:
    """Compare blockIDs from overlaps.h5 and from hamiltonians.h5."""

    def get_block_identitys_dict(filename: str) -> Dict[Tuple[int, int], Tensor]:
        # 从element.dat中读取元素类型信息，得到atom_types 
        # atom_types: 记录体系中各个原子的元素类型，即原子序数
        atom_types: List[int] = []  

        file: str = os.path.join(folder, 'element.dat')
        with open(file) as f:
            line = f.readline()
            while line:
                atomic_number: int = int(line.split()[0])
                atom_types.append(atomic_number)
                line = f.readline()

        # 只留下atom_types
        del file, f, line, atomic_number 


        # 从hamiltonians.h5得到block_identitys_dict
        # blocks_dict: 若干分块矩阵，按照原子对种类分类
        # block_identitys_dict：blocks_dict中分块矩阵所对应的原子对(p, q, Rx, Ry, Rz)，按照原子对种类分类
        block_identitys_dict: Dict[Tuple[int, int], Tensor] = {}

        file = os.path.join(folder, filename)
        f = h5py.File(file, 'r')
        for key, _ in f.items():
            key = json.loads(key)
            p = key[3] - 1 # hamiltonians.h5中的原子编号是1-based的
            q = key[4] - 1
            Rx: int = key[0] # p, q原子所在晶胞的相对位置
            Ry: int = key[1]
            Rz: int = key[2]

            p_type = atom_types[p]
            q_type = atom_types[q]
            pair_type = get_valid_pair_type(p_type, q_type)
            blockID = torch.tensor([p, q, Rx, Ry, Rz], dtype=torch.int)

            if pair_type not in block_identitys_dict:
                block_identitys_dict[pair_type] = [blockID]
            else:
                block_identitys_dict[pair_type].append(blockID)

        
        # now variable blockIDs is List[Tensor]
        # change it to Tensor here
        for pair_type, blockIDs in block_identitys_dict.items():
            blockIDs = torch.stack(blockIDs, dim=0)
            block_identitys_dict[pair_type] = blockIDs
        
        return block_identitys_dict

    from_H_dict: Dict[Tuple[int, int], Tensor] = get_block_identitys_dict(filename='hamiltonians.h5')
    from_S_dict: Dict[Tuple[int, int], Tensor] = get_block_identitys_dict(filename='overlaps.h5')


    for keyH in from_H_dict.keys():

        if keyH not in from_S_dict.keys():
            logger.info(f"keyH={keyH} not in from_S_dict.keys()")
            return False
        
        valueH: List[List[int]] = from_H_dict[keyH].tolist()
        valueS: List[List[int]] = from_S_dict[keyH].tolist()
        valueH.sort()
        valueS.sort()
        if valueH!=valueS:
            logger.info(f"valueH=from_H_dict[{keyH}] not equal to valueS=from_S_dict[{keyH}]")
            return False


    for keyS in from_S_dict.keys():

        if keyS not in from_H_dict.keys():
            logger.info(f"keyS={keyS} not in from_H_dict.keys()")
            return False
        
        valueH: List[List[int]] = from_H_dict[keyS].tolist()
        valueS: List[List[int]] = from_S_dict[keyS].tolist()
        valueH.sort()
        valueS.sort()
        if valueH!=valueS:
            logger.info(f"valueH=from_H_dict[{keyS}] not equal to valueS=from_S_dict[{keyS}]")
            return False

    return True


# Train Process Helper in main.py
def revert_blocks_dict(out_dict: Dict[Tuple[str], Tensor], 
        sort_dict: Dict[Tuple[int, int], List[int]],
        Ls_dict: Dict[int, List[int]], 
        is_spin: bool,
        default_dtype_torch: torch.dtype, 
        device: str
        ) -> Dict[Tuple[int, int], Tensor]:
    """Revert from output of GNN model 
    `(hamiltonian block in direct sum & simplified Irreps form)`
    to blocks_dict 
    `(origin hamiltonian block in tensor product & origin Irreps form)`.

    if is_spin=True, 
    Tensor shape: `(num_blocks, 2, 4, p_num_basis, q_num_basis)`
    Note that it is in 2 parts: real, imag, and the element is not complex number.
    Note that the real part comes first in dimension 0, then comes the imag part.

    if is_spin=False,
    Tensor shape: `(num_blocks, p_num_basis, q_num_basis)`
    """
    blocks_dict: Dict[Tuple[int, int], Tensor] = {}

    for pair_type, irreps in out_dict.items():
        # ================0.得到矩阵切片================
        # 从Ls_dict得到（当前原子对类型下）原子基函数角动量信息
        # 我们认为相同元素类型的原子基函数角动量信息完全相同, 所以从第一个atom_pair得到Ls
        pair_type = to_dtype_edge(pair_type, dtype=[Tuple, int]) # change "83-83" to (83, 83)
        p, q = pair_type 
        p_Ls: NDArray[np.int] = np.array(Ls_dict[p])
        q_Ls: NDArray[np.int] = np.array(Ls_dict[q])
        p_num_basis_group: int = len(p_Ls)
        q_num_basis_group: int = len(q_Ls)
        p_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * p_Ls + 1), obj=0, values=0) # 往第一个位置插入0
        q_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * q_Ls + 1), obj=0, values=0)
        L_tuples: List[Tuple[int, int]] = []
        tuple_slices: List[int] = [0]
        # 从p_Ls, q_Ls得到上一个阶段矩阵irreps中按照特定顺序排布的一维基函数组对分块矩阵的irreps中各个基函数组的角动量对L_tuples
        # 先p后q，顺序和下面一样
        for p_single_L in p_Ls:
            for q_single_L in q_Ls:
                L_tuples.append(tuple([int(p_single_L), int(q_single_L)])) # 若不int()就是np.int64, 报错, out_js_list中的整数必须是python的int 
                num_basis_L_tuple: int = (1 + 2 * int(p_single_L)) * (1 + 2 * int(q_single_L))
                tuple_slices.append(tuple_slices[-1] + num_basis_L_tuple)
        
        # ================2.重排矩阵元顺序，按照原重叠矩阵Irreps复原================
        sort_idx: List[int] = sort_dict[pair_type]
        # if is_spin=True:
        #(before e3TensorDecomp) irreps (real): num_blocks x (2 x 4 x p_num_basis x q_num_basis)
        # if is_spin=False:
        # irreps (real): num_blocks x (p_num_basis x q_num_basis)
        # Note that len(sort_idx) == (2 x 4 x p_num_basis x q_num_basis) or (p_num_basis x q_num_basis)
        irreps: Tensor = irreps[:, sort_idx]

        # ================2.笛卡尔坐标->球谐坐标的坐标变换================
        transformer = e3TensorDecomp(
            out_js_list=L_tuples, 
            default_dtype_torch=default_dtype_torch, 
            spinful=is_spin, 
            device_torch=device
            ) 
        irreps = transformer.get_H(irreps)
        # 只留下tuple_slices, p_num_basis_group, q_num_basis_group, p_slices, q_slices (和blocks_dict, irreps, pair_type)
        del p, q, p_Ls, q_Ls, p_single_L, q_single_L, 
        del transformer, num_basis_L_tuple, L_tuples

        # ================3.改变分块矩阵形状重新排序================
        # 从原子对分块矩阵blocks借助p_slices, q_slices，tuple_slices得到原始二维基函数组对分块矩阵的blocks
        num_blocks: int = irreps.shape[0] # number of blocks of the current atom pair type
        if is_spin==True:
            blocks: Tensor = torch.zeros((num_blocks, 4, p_slices[-1], q_slices[-1]), dtype=flt2cplx(default_dtype_torch), device=device) 
        else:
            blocks: Tensor = torch.zeros((num_blocks, p_slices[-1], q_slices[-1]), dtype=default_dtype_torch, device=device)
        #Bug still only cpu: blocks.to(device=device)
        
        # 分别对原子p和q的不可约基函数组i和j循环，先p后q与先q后p可以导致完全不同顺序排布
        for i in range(p_num_basis_group): 
            i_slice = slice(p_slices[i], p_slices[i+1]) # p_slices的长度为（1+p_num_basis_group）
            i_len: int = i_slice.stop - i_slice.start
            for j in range(q_num_basis_group):
                j_slice = slice(q_slices[j], q_slices[j+1]) # q_slices的长度为（1+q_num_basis_group）
                j_len: int = j_slice.stop - j_slice.start
                k = j + i * q_num_basis_group
                k_slice = slice(tuple_slices[k], tuple_slices[k+1])

                # reshape规则不同的话也可以导致完全不同顺序排布
                if is_spin==True:
                    # irreps (complex): num_blocks x 4 x (p_num_basis x q_num_basis)
                    # current blocks (complex): num_blocks x 4 x p_num_basis x q_num_basis
                    blocks[:, :, i_slice, j_slice] = irreps[:, :, k_slice].reshape(num_blocks, 4, i_len, j_len)
                    # bug code for test
                    # blocks[:, :, j_slice, i_slice] = irreps[:, :, k_slice].reshape(num_blocks, 4, j_len, i_len) 
                else:
                    # irreps (real): num_blocks x (p_num_basis x q_num_basis)
                    # blocks (real): num_blocks x p_num_basis x q_num_basis
                    blocks[:, i_slice, j_slice] = irreps[:, k_slice].reshape(num_blocks, i_len, j_len)

        # 只留下blocks, (和blocks_dict, pair_type)
        del tuple_slices, p_num_basis_group, q_num_basis_group, p_slices, q_slices
        del irreps, num_blocks, i, j, i_slice, j_slice, k, k_slice, i_len, j_len


        # ================4.给blocks_dict赋值================
        if is_spin==True:
            # = break complex into real, imag parts = 
            #Before Tensor shape:  (num_blocks, 4, p_num_basis, q_num_basis)
            #After Tensor shape: (num_blocks, 2, 4, p_num_basis, q_num_basis)
            blocks_dict[pair_type] = torch.stack([blocks.real, blocks.imag], dim=1)
        else:
            # blocks (real): num_blocks x p_num_basis x q_num_basis
            blocks_dict[pair_type] = blocks
    
    return blocks_dict


def get_loss(criterion: torch.nn.modules.loss._Loss,
        y_dict: Dict[Tuple[str, str, str], Tensor],
        out_dict: Dict[str, Tensor], # TODO
        sort_dict: Dict[Tuple[int, int], List[int]],
        Ls_dict: Dict[int, List[int]], 
        is_spin: bool,
        default_dtype_torch: torch.dtype, 
        device: str) -> Tensor:
    """Get Loss.
    
    Notice the data type of pair_type: \n
    out_dict (predicted hamiltonian in direct sum form and sorted simple irreps form
            from model forward function): `str` \n
    output (predicted hamiltonian in tensor product form and original irreps form
            return value from revert_blocks_dict function): `Tuple[int, int]` \n
    y_dict (true hamiltonian in tensor product form and original irreps form
            value read from HeteroDataset or pytorch DataLoader): `Tuple[str, str, str]`
    """
    output: Dict[Tuple[int, int], Tensor] = revert_blocks_dict(
        out_dict=out_dict,
        sort_dict=sort_dict,
        Ls_dict=Ls_dict, 
        is_spin=is_spin,
        default_dtype_torch=default_dtype_torch, 
        device=device
    )
    loss: Tensor = torch.tensor(0, dtype=default_dtype_torch, device=device)
    for edge, y_hat in output.items(): 
        # edge here is from revert_blocks_dict, so is Tuple[int, int] actually.
        # we need change -- Tuple[int, int] -> Tuple[str, str, str] here
        # (83, 83) to ('83', '83-83', '83')
        #Bug edge: '83-83' to ('83', '83-83', '83')
        #Bug edge: Tuple[str, str, str] = (edge.split('-')[0], edge, edge.split('-')[1])
        edge: Tuple[str, str, str] = to_dtype_edge(edge, dtype=[Tuple, str])
        y = y_dict[edge]
        loss = loss + criterion(y_hat, y) # .to(device=config.device)
    loss = loss / len(output) # number of key-value in output dictionary
    return loss


# = Read Information from OpenMX = 
# Be Careful when changing the codes !
# For example, These Reading-Block-Codes is copied from each other, 
#   so if one is changed, others should alse be changed: 
#   read_x_dict_from_one_OpenMX_folder,
#   read_label_dict_from_one_OpenMX_folder
def old_0_read_label_dict_from_one_OpenMX_folder(folder: str, is_spin: bool, default_dtype_torch: torch.dtype) -> \
        Dict[Tuple[int, int], Tensor]:
    """Read from hamiltonians.h5 and get label_dict (hamiltonian matrix in direct sum form)."""

    # 从element.dat中读取元素类型信息，得到atom_types 
    # atom_types: 记录体系中各个原子的元素类型，即原子序数
    atom_types: List[int] = []  

    file: str = os.path.join(folder, 'element.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            atomic_number: int = int(line.split()[0])
            atom_types.append(atomic_number)
            line = f.readline()

    # 只留下atom_types
    del file, f, line, atomic_number 




    
    # 从hamiltonians.h5借助atom_types读取H，得到blocks_dict和block_identitys_dict
    # blocks_dict: 若干分块矩阵，按照原子对种类分类
    # block_identitys_dict：blocks_dict中分块矩阵所对应的原子对(p, q, Rx, Ry, Rz)，按照原子对种类分类
    blocks_dict: Dict[Tuple[int, int], Tensor] = {}
    block_identitys_dict: Dict[Tuple[int, int], Tensor] = {}

    #TODO different from read_x_dict_from_one_OpenMX_folder here
    file = os.path.join(folder, 'hamiltonians.h5')
    f = h5py.File(file, 'r')
    for key, block in f.items():
        key = json.loads(key)
        p = key[3] - 1 # hamiltonians.h5中的原子编号是1-based的
        q = key[4] - 1
        Rx: int = key[0] # p, q原子所在晶胞的相对位置
        Ry: int = key[1]
        Rz: int = key[2]

        p_type = atom_types[p]
        q_type = atom_types[q]
        pair_type = get_valid_pair_type(p_type, q_type)

        block = np.array(block)
        
        #TODO S spin
        #TODO different from read_x_dict_from_one_OpenMX_folder here 
        if is_spin is True:
            block = torch.from_numpy(block).detach().clone().to(flt2cplx(default_dtype_torch))
            p_num_basis: float = block.shape[0] / 2
            q_num_basis: float = block.shape[1] / 2
            # assert p_num_basis.is_integer() and q_num_basis.is_integer() #TODO just test
            p_num_basis: int = int(p_num_basis)
            q_num_basis: int = int(q_num_basis)
            # break the origin block into 4 piece according to spin of 2 atoms(upxup, upxdown, downxup, downxdown) and stack
            #Before:(block matrix style)
            # block=[ [p_up x q_up, p_up x q_down],
            #         [p_down x q_up, p_down x q_down]
            #       ]
            #After:(block matrix style)
            # block=[p_up x q_up, p_up x q_down, p_down x q_up, p_down x q_down]
            # block shape change: (2 x p_num_basis) x (2 x q_num_basis) -> 4 x p_num_basis x q_num_basis
            block = torch.stack([
                block[:p_num_basis, :q_num_basis], block[:p_num_basis, q_num_basis:],
                block[p_num_basis:, :q_num_basis], block[p_num_basis:, q_num_basis:]
            ], dim=0)

        else:
            block = torch.from_numpy(block).detach().clone().to(default_dtype_torch)


        blockID = torch.tensor([p, q, Rx, Ry, Rz], dtype=torch.int)

        if pair_type not in blocks_dict:
            blocks_dict[pair_type] = [block]
            block_identitys_dict[pair_type] = [blockID]
        else:
            blocks_dict[pair_type].append(block)
            block_identitys_dict[pair_type].append(blockID)

    # now variable blocks is List[Tensor]
    # change it to Tensor here
    for pair_type, blocks in blocks_dict.items():
        blocks = torch.stack(blocks, dim=0)
        blocks_dict[pair_type] = blocks
    
    # now variable blockIDs is List[Tensor]
    # change it to Tensor here
    for pair_type, blockIDs in block_identitys_dict.items():
        blockIDs = torch.stack(blockIDs, dim=0)
        block_identitys_dict[pair_type] = blockIDs
    
    # 只留下blocks_dict和block_identitys_dict
    del file, f, key, p, q, Rx, Ry, Rz, p_type, q_type, pair_type, block, blocks, blockID, blockIDs 
    del atom_types




    #TODO different from read_x_dict_from_one_OpenMX_folder here
    # 这里的edge_index_dict是从hamiltonians.h5中读取的，和从overlaps.h5中读取的不一样，暂时留住这部分代码注释起来
    """
    # 从block_identitys_dict得到edge_index_dict
    edge_index_dict: Dict[Tuple[int, int], Tensor] = {}
    for pair_type, blockIDs in block_identitys_dict.items():
        # blockIDs: Tensor in the shape of (num_blockIDs, 5), (p, q, r_box_x, r_box_y, r_box_z) in the current atom pair type (p_type, q_type)
        edge_indexs: Tensor = torch.transpose(blockIDs[:, [0, 1]], 0, 1).contiguous()
        # edge_indexs: Tensor in the shape of (2, num_edges) Note that num_edges=num_blocks=num_blockIDs
        edge_index_dict[pair_type] = edge_indexs

    # 只留下edge_index_dict(和blocks_dict, block_identitys_dict)
    del pair_type, blockIDs, edge_indexs
    """
    




    # 从type_orbitals.dat读取各原子的基函数角动量信息，得到Ls_list
    Ls_list: List[List[int]] = []
    file = os.path.join(folder, 'orbital_types.dat')

    with open(file) as f:
        line = f.readline()
        while line:
            Ls = [int(x) for x in line.split()] 
            # Ls: basis_group_Ls, angular momentum list for all basis group of the current atom
            Ls_list.append(Ls)
            line = f.readline()
    
    # 只留下Ls_list(和blocks_dict, block_identitys_dict)
    del file, f, line, Ls




    #TODO different from read_x_dict_from_one_OpenMX_folder here
    # 从blocks_dict借助blocks_identity_dict, Ls_list得到edge_label_dict
    edge_label_dict: Dict[Tuple[int, int], Tensor] = {}

    for pair_type, blockIDs in block_identitys_dict.items():
        
        
        # 从Ls_list借助blockIDs得到（当前原子对类型下）原子基函数角动量信息
        # 我们认为相同元素类型的原子基函数角动量信息完全相同, 所以从第一个atom_pair得到Ls
        atom_pair: Tuple[int, int] = blockIDs[0, [0, 1]]
        p = atom_pair[0]
        q = atom_pair[1]
        p_Ls: NDArray[np.int] = np.array(Ls_list[p])
        q_Ls: NDArray[np.int] = np.array(Ls_list[q])
        p_num_basis_group: int = len(p_Ls)
        q_num_basis_group: int = len(q_Ls)
        p_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * p_Ls + 1), obj=0, values=0) # 往第一个位置插入0
        q_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * q_Ls + 1), obj=0, values=0)


        # ================1.改变分块矩阵形状重新排序================
        # 从原子对分块矩阵blocks借助p_slices, q_slices得到按照特定顺序排布的一维基函数组对分块矩阵的irreps(blocks的替代者)
        irreps: Tensor = [] # blocks的替代者
        blocks: Tensor = blocks_dict[pair_type]
        num_blocks: int = blocks.shape[0] # number of blocks of the current atom pair type
        
        # 分别对原子p和q的不可约基函数组i和j循环，先p后q与先q后p可以导致完全不同顺序排布的一维不可约分块矩阵的blocks_dict
        for i in range(p_num_basis_group): 
            i_slice = slice(p_slices[i], p_slices[i+1]) # p_slices的长度为（1+p_num_basis_group）
            for j in range(q_num_basis_group):
                j_slice = slice(q_slices[j], q_slices[j+1]) # q_slices的长度为（1+q_num_basis_group）

                # reshape规则不同的话也可以导致完全不同顺序排布的一维不可约分块矩阵的blocks_dict
                #TODO S spin
                if is_spin==True:
                    # blocks: num_blocks x 4 x p_num_basis x q_num_basis
                    single_irrep: Tensor = blocks[:, :, i_slice, j_slice].reshape(num_blocks, 4, -1)
                else:
                    # blocks: num_blocks x  p_num_basis x q_num_basis
                    single_irrep: Tensor = blocks[:, i_slice, j_slice].reshape(num_blocks, -1)

                irreps.append(single_irrep)
        
        # List of Tensor to Tensor
        irreps = torch.cat(irreps, dim=-1)

        # 只留下irreps, p_Ls, q_Ls, pair_type(和blocks_dict, block_identitys_dict)
        del blockIDs, atom_pair, p, q, p_num_basis_group, q_num_basis_group, p_slices, q_slices
        del blocks, num_blocks, i, j, i_slice, j_slice, single_irrep



        # ================2.笛卡尔坐标->球谐坐标的坐标变换================
        L_tuples: List[Tuple[int, int]] = []
        # 从p_Ls, q_Ls得到上一个阶段矩阵irreps中按照特定顺序排布的一维基函数组对分块矩阵的irreps中各个基函数组的角动量对L_tuples
        # 先p后q，顺序一定要保持和上面一样
        for p_single_L in p_Ls:
            for q_single_L in q_Ls:
                L_tuples.append(tuple([int(p_single_L), int(q_single_L)])) # 若不int()就是np.int64, 报错, out_js_list中的整数必须是python的int 
        
        transformer = e3TensorDecomp(out_js_list=L_tuples, 
                                        default_dtype_torch=default_dtype_torch, 
                                        spinful=is_spin) #TODO S spin
        irreps = transformer.get_net_out(irreps)


        # 只留下irreps, pair_type(和blocks_dict, block_identitys_dict)
        del L_tuples, p_Ls, q_Ls, p_single_L, q_single_L, transformer

        #TODO different from read_x_dict_from_one_OpenMX_folder here
        # ================3.给edge_label_dict赋值================
        
        edge_label_dict[pair_type] = irreps


    return edge_label_dict


def read_x_dict_from_one_OpenMX_folder(folder: str, is_spin: bool, default_dtype_torch: torch.dtype):
    """Read from overlaps.h5 and get node_fea_dict, edge_index_dict, edge_fea_dict."""
    ###############################################################
    ###############################################################
    # 从element.dat中读取元素类型信息，得到atom_types和node_fea_dict
    # atom_types: 记录体系中各个原子的元素类型，即原子序数
    atom_types: List[int] = []  

    file: str = os.path.join(folder, 'element.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            atomic_number: int = int(line.split()[0])
            atom_types.append(atomic_number)
            line = f.readline()

    # 只留下atom_types
    del file, f, line, atomic_number 

    ###############################################################
    ###############################################################
    # 从atom_types中得到node_fea_dict
    node_fea_dict: Dict[int, Tensor] = {}

    types: Set[int] = set(atom_types)
    for atomic_number in types:
        size: Tuple[int] = tuple([atom_types.count(atomic_number)])
        node_fea_dict[atomic_number] = torch.full(size=size, fill_value=atomic_number)
    r"""
        node_fea_dict[atomic_number] = \
                one_hot(torch.full(size=size, fill_value=atomic_number), num_classes=118)"""
    # 只留下node_fea_dict(和atom_types)
    del types, atomic_number, size
    
    ###############################################################
    ###############################################################
    # 从overlaps.h5借助atom_types读取H，得到blocks_dict和block_identitys_dict
    # blocks_dict: 若干分块矩阵，按照原子对种类分类
    # block_identitys_dict：blocks_dict中分块矩阵所对应的原子对(p, q, Rx, Ry, Rz)，按照原子对种类分类
    blocks_dict: Dict[Tuple[int, int], Tensor] = {}
    block_identitys_dict: Dict[Tuple[int, int], Tensor] = {}

    file = os.path.join(folder, 'overlaps.h5')
    f = h5py.File(file, 'r')
    for key, block in f.items():
        key = json.loads(key)
        p = key[3] - 1 # hamiltonians.h5中的原子编号是1-based的
        q = key[4] - 1
        Rx: int = key[0] # p, q原子所在晶胞的相对位置
        Ry: int = key[1]
        Rz: int = key[2]

        p_type = atom_types[p]
        q_type = atom_types[q]
        pair_type = get_valid_pair_type(p_type, q_type)

        block = np.array(block)
        """
        #just test
        flag = 0
        if flag==0:
            logger.info(f"block.dtype={block.dtype}")
            flag+=1
        """
        
        #TODO S spin
        if is_spin is True:
            block = torch.from_numpy(block).detach().clone().to(flt2cplx(default_dtype_torch))
            block = block + 1j * block
            block = block.repeat((4, 1, 1))
        else:
            block = torch.from_numpy(block).detach().clone().to(default_dtype_torch)


        blockID = torch.tensor([p, q, Rx, Ry, Rz], dtype=torch.int)

        if pair_type not in blocks_dict:
            blocks_dict[pair_type] = [block]
            block_identitys_dict[pair_type] = [blockID]
        else:
            blocks_dict[pair_type].append(block)
            block_identitys_dict[pair_type].append(blockID)

    # now variable blocks is List[Tensor]
    # change it to Tensor here
    for pair_type, blocks in blocks_dict.items():
        blocks = torch.stack(blocks, dim=0)
        blocks_dict[pair_type] = blocks
    
    # now variable blockIDs is List[Tensor]
    # change it to Tensor here
    for pair_type, blockIDs in block_identitys_dict.items():
        blockIDs = torch.stack(blockIDs, dim=0)
        block_identitys_dict[pair_type] = blockIDs
    
    # 只留下blocks_dict和block_identitys_dict(和node_fea_dict, atom_types)
    del file, f, key, p, q, Rx, Ry, Rz, p_type, q_type, pair_type, block, blocks, blockID, blockIDs 

    # 2023/12/31 deal with bug-edge_index for HeteroData (global index -> partial index)
    atom_types: NDArray = np.array(atom_types) # shape=(num_atoms, )
    partial_index: NDArray = np.empty_like(atom_types) # shape=(num_atoms, )
    # partial_index: (global index -> partial index)
    for atomic_number in np.unique(atom_types):
        g: NDArray = np.where(atom_types==atomic_number)[0]
        partial_index[g] = np.arange(len(g))
    partial_index: Tensor = torch.from_numpy(partial_index)
    del atom_types, g, atomic_number

    ###############################################################
    ###############################################################
    # 从block_identitys_dict得到edge_index_dict
    edge_index_dict: Dict[Tuple[int, int], Tensor] = {}
    for pair_type, blockIDs in block_identitys_dict.items():
        # blockIDs: Tensor in the shape of (num_blockIDs, 5), (p, q, r_box_x, r_box_y, r_box_z) in the current atom pair type (p_type, q_type)
        global_edge_indexs: Tensor = torch.transpose(blockIDs[:, [0, 1]], 0, 1).contiguous()
        # global_edge_indexs: Tensor in the shape of (2, num_edges) Note that num_edges=num_blocks=num_blockIDs
        #(Notice !) This is not valid edge_indexs of HeteroData ! 
        #(Notice !) Pair_type=(src, dst)'s edge_index should be index of x_dict[src], x_dict[dst] !
        edge_indexs: Tensor = partial_index[global_edge_indexs]
        # edge_indexs: Tensor in the shape of (2, num_edges) Note that num_edges=num_blocks=num_blockIDs
        edge_index_dict[pair_type] = edge_indexs

    # 只留下edge_index_dict(和blocks_dict, block_identitys_dict, node_fea_dict)
    del pair_type, blockIDs, edge_indexs, global_edge_indexs
    del partial_index

    ###############################################################
    ###############################################################
    # 从site_postitions和lat.dat读取原子笛卡尔坐标和晶格矢量，得到positions，lattice_vectors，
    # 再利用block_identitys_dict，得到只包含相对位置信息的edge_fea_dict
    # 只包含相对位置信息的edge_fea_dict: 
    edge_fea_dict: Dict[Tuple[int, int], Tensor] = {}

    positions = np.loadtxt(os.path.join(folder, 'site_positions.dat')).T
    positions = torch.tensor(positions, dtype=default_dtype_torch)
    lattice_vectors = np.loadtxt(os.path.join(folder, 'lat.dat')).T
    lattice_vectors = torch.tensor(lattice_vectors, dtype=default_dtype_torch)

    for pair_type, blockIDs in block_identitys_dict.items():
        atom_pairs: Tensor = blockIDs[:, [0, 1]]
        lattice_locations: Tensor = blockIDs[:, [2, 3, 4]]
        p_positions = positions[atom_pairs[:, 0]]
        q_positions = positions[atom_pairs[:, 1]] \
                    + lattice_locations.type(default_dtype_torch) @ lattice_vectors
        distances = torch.linalg.vector_norm(q_positions - p_positions, dim=-1)
        position_vectors = q_positions - p_positions

        # edge_fea_positions[pair_type]: Tensor in the shape of (num_blocks, 4)
        edge_fea_dict[pair_type] = torch.cat([distances[:, None], position_vectors], dim=-1)

    # 只留下只包含相对位置信息的edge_fea_dict(和edge_index_dict, blocks_dict, block_identitys_dict, node_fea_dict)
    del positions, lattice_vectors, pair_type, blockIDs, atom_pairs, lattice_locations, p_positions, q_positions, distances, position_vectors

    ###############################################################
    ###############################################################
    # 从type_orbitals.dat读取各原子的基函数角动量信息，得到Ls_list
    Ls_list: List[List[int]] = []
    file = os.path.join(folder, 'orbital_types.dat')

    with open(file) as f:
        line = f.readline()
        while line:
            Ls = [int(x) for x in line.split()] 
            # Ls: basis_group_Ls, angular momentum list for all basis group of the current atom
            Ls_list.append(Ls)
            line = f.readline()
    
    # 只留下 Ls_list(和只包含相对位置信息的edge_fea_dict, edge_index_dict, blocks_dict, block_identitys_dict, node_fea_dict)
    del file, f, line, Ls

    ###############################################################
    ###############################################################
    # 从blocks_dict借助blocks_identity_dict, Ls_list和只包含相对位置信息的edge_fea_dict得到edge_fea_dict
    for pair_type, blockIDs in block_identitys_dict.items():
        # 从Ls_list借助blockIDs得到（当前原子对类型下）原子基函数角动量信息
        # 我们认为相同元素类型的原子基函数角动量信息完全相同, 所以从第一个atom_pair得到Ls
        atom_pair: Tuple[int, int] = blockIDs[0, [0, 1]]
        p = atom_pair[0]
        q = atom_pair[1]
        p_Ls: NDArray[np.int] = np.array(Ls_list[p])
        q_Ls: NDArray[np.int] = np.array(Ls_list[q])
        p_num_basis_group: int = len(p_Ls)
        q_num_basis_group: int = len(q_Ls)
        p_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * p_Ls + 1), obj=0, values=0) # 往第一个位置插入0
        q_slices: NDArray[np.int] = np.insert(arr=np.cumsum(2 * q_Ls + 1), obj=0, values=0)

        # ================1.改变分块矩阵形状重新排序================
        # 从原子对分块矩阵blocks借助p_slices, q_slices得到按照特定顺序排布的一维基函数组对分块矩阵的irreps(blocks的替代者)
        irreps: Tensor = [] # blocks的替代者
        blocks: Tensor = blocks_dict[pair_type]
        num_blocks: int = blocks.shape[0] # number of blocks of the current atom pair type
        
        # 分别对原子p和q的不可约基函数组i和j循环，先p后q与先q后p可以导致完全不同顺序排布的一维不可约分块矩阵的blocks_dict
        for i in range(p_num_basis_group): 
            i_slice = slice(p_slices[i], p_slices[i+1]) # p_slices的长度为（1+p_num_basis_group）
            for j in range(q_num_basis_group):
                j_slice = slice(q_slices[j], q_slices[j+1]) # q_slices的长度为（1+q_num_basis_group）

                # reshape规则不同的话也可以导致完全不同顺序排布的一维不可约分块矩阵的blocks_dict
                #TODO S spin
                if is_spin==True:
                    # blocks: num_blocks x 4 x p_num_basis x q_num_basis
                    single_irrep: Tensor = blocks[:, :, i_slice, j_slice].reshape(num_blocks, 4, -1)
                else:
                    # blocks: num_blocks x  p_num_basis x q_num_basis
                    single_irrep: Tensor = blocks[:, i_slice, j_slice].reshape(num_blocks, -1)

                irreps.append(single_irrep)
        
        # List of Tensor to Tensor
        irreps = torch.cat(irreps, dim=-1)

        # 只留下irreps, p_Ls, q_Ls, pair_type(和只包含相对位置信息的edge_fea_dict, edge_index_dict, blocks_dict, block_identitys_dict, node_fea_dict)
        del blockIDs, atom_pair, p, q, p_num_basis_group, q_num_basis_group, p_slices, q_slices
        del blocks, num_blocks, i, j, i_slice, j_slice, single_irrep

        # ================2.笛卡尔坐标->球谐坐标的坐标变换================
        L_tuples: List[Tuple[int, int]] = []
        # 从p_Ls, q_Ls得到上一个阶段矩阵irreps中按照特定顺序排布的一维基函数组对分块矩阵的irreps中各个基函数组的角动量对L_tuples
        # 先p后q，顺序一定要保持和上面一样
        for p_single_L in p_Ls:
            for q_single_L in q_Ls:
                L_tuples.append(tuple([int(p_single_L), int(q_single_L)])) # 若不int()就是np.int64, 报错, out_js_list中的整数必须是python的int 
        
        transformer = e3TensorDecomp(out_js_list=L_tuples, 
                                        default_dtype_torch=default_dtype_torch, 
                                        spinful=is_spin) #TODO S spin
        irreps = transformer.get_net_out(irreps)
        """TODO S spin
        if is_spin:
            num_blocks: int = batch_irreps.shape[0]
            batch_irreps: Tensor = batch_irreps.reshape(num_blocks, 4, -1)
            logger.info(f"batch_irreps.shape = {batch_irreps.shape}") # just test
        else:
            pass
        coupled_batch_irreps: Tensor = transformer.get_net_out(batch_irreps)
        """
        # 只留下irreps, pair_type(和只包含相对位置信息的edge_fea_dict, edge_index_dict, blocks_dict, block_identitys_dict, node_fea_dict)
        del L_tuples, p_Ls, q_Ls, p_single_L, q_single_L, transformer


        # ================3.更新只包含相对位置信息的edge_fea_dict================
        edge_fea: Tensor = edge_fea_dict[pair_type] # shape=(num_blocks, 4)
        
        edge_fea_dict[pair_type] = torch.cat([edge_fea, irreps], dim=-1) # shape=(num_blocks, 4 + (p_num_basis x q_num_basis)) TODO S spin


    return node_fea_dict, edge_index_dict, edge_fea_dict


def read_label_dict_from_one_OpenMX_folder(folder: str, is_spin: bool, default_dtype_torch: torch.dtype) -> \
        Dict[Tuple[int, int], Tensor]:
    """Read from hamiltonians.h5 and get blocks_dict 
    `(origin hamiltonian block in tensor product form)`.

    if is_spin=True, 
    Tensor shape: `(num_blocks, 2, 4, p_num_basis, q_num_basis)`
    Note that it is in 2 parts: real, imag, and the element is not complex number.
    Note that the real part comes first in dimension 0, then comes the imag part.

    if is_spin=False,
    Tensor shape: `(num_blocks, p_num_basis, q_num_basis)`
    """
    # 从element.dat中读取元素类型信息，得到atom_types 
    # atom_types: 记录体系中各个原子的元素类型，即原子序数
    atom_types: List[int] = []  

    file: str = os.path.join(folder, 'element.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            atomic_number: int = int(line.split()[0])
            atom_types.append(atomic_number)
            line = f.readline()

    # 只留下atom_types
    del file, f, line, atomic_number 




    
    # 从hamiltonians.h5借助atom_types读取H，得到blocks_dict
    # blocks_dict: 若干分块矩阵，按照原子对种类分类
    blocks_dict: Dict[Tuple[int, int], Tensor] = {}

    file = os.path.join(folder, 'hamiltonians.h5')
    f = h5py.File(file, 'r')
    for key, block in f.items():
        key = json.loads(key)
        p = key[3] - 1 # hamiltonians.h5中的原子编号是1-based的
        q = key[4] - 1
        Rx: int = key[0] # p, q原子所在晶胞的相对位置
        Ry: int = key[1]
        Rz: int = key[2]

        p_type = atom_types[p]
        q_type = atom_types[q]
        pair_type = get_valid_pair_type(p_type, q_type)

        block = np.array(block)
        
        #TODO S spin
        #TODO different from read_x_dict_from_one_OpenMX_folder here 
        if is_spin is True:
            block = torch.from_numpy(block).detach().clone().to(flt2cplx(default_dtype_torch))
            p_num_basis: float = block.shape[0] / 2
            q_num_basis: float = block.shape[1] / 2
            # assert p_num_basis.is_integer() and q_num_basis.is_integer() # just test
            p_num_basis: int = int(p_num_basis)
            q_num_basis: int = int(q_num_basis)
            # break the origin block into 4 piece according to spin of 2 atoms(upxup, upxdown, downxup, downxdown) and stack
            #Before:(block matrix style)
            # block=[ [p_up x q_up, p_up x q_down],
            #         [p_down x q_up, p_down x q_down]
            #       ]
            #After:(block matrix style)
            # block=[p_up x q_up, p_up x q_down, p_down x q_up, p_down x q_down]
            # block shape change: (2 x p_num_basis) x (2 x q_num_basis) -> 4 x p_num_basis x q_num_basis
            block = torch.stack([
                block[:p_num_basis, :q_num_basis], block[:p_num_basis, q_num_basis:],
                block[p_num_basis:, :q_num_basis], block[p_num_basis:, q_num_basis:]
            ], dim=0)

        else:
            block = torch.from_numpy(block).detach().clone().to(default_dtype_torch)


        if pair_type not in blocks_dict:
            blocks_dict[pair_type] = [block]
        else:
            blocks_dict[pair_type].append(block)

    # now variable blocks is List[Tensor]
    # change it to Tensor here
    for pair_type, blocks in blocks_dict.items():
        blocks = torch.stack(blocks, dim=0)
        if is_spin==True:
            # = break complex into real, imag parts = 
            #Before Tensor shape:  (num_blocks, 4, p_num_basis, q_num_basis)
            #After Tensor shape: (num_blocks, 2, 4, p_num_basis, q_num_basis)
            blocks_dict[pair_type] = torch.stack([blocks.real, blocks.imag], dim=1)
        else:
            blocks_dict[pair_type] = blocks
    
    # 只留下blocks_dict和block_identitys_dict
    del file, f, key, p, q, Rx, Ry, Rz, p_type, q_type, pair_type, block, blocks
    del atom_types


    return blocks_dict


def read_Ls_dict_from_one_OpenMX_folder(folder: str) -> Dict[int, List[int]]:
    """Get Ls_dict from one OpenMX output folder."""
    # 从element.dat中读取元素类型信息，得到atom_types 
    # atom_types: 记录体系中各个原子的元素类型，即原子序数
    atom_types: List[int] = []  
    file: str = os.path.join(folder, 'element.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            atomic_number: int = int(line.split()[0])
            atom_types.append(atomic_number)
            line = f.readline()
    # 只留下atom_types
    del file, f, line, atomic_number 

    # 从type_orbitals.dat读取各原子的基函数角动量信息，得到Ls_dict
    Ls_list: List[List[int]] = []
    file = os.path.join(folder, 'orbital_types.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            Ls = [int(x) for x in line.split()] 
            # Ls: basis_group_Ls, angular momentum list for all basis group of the current atom
            Ls_list.append(Ls)
            line = f.readline()
    # 只留下Ls_list(和atom_types)
    del file, f, line, Ls

    # 构造Ls_dict，每种atom_type的Ls_dict只记录一次
    Ls_dict: Dict[int, List[int]] = {}
    set_atom_types: Set[int] = set(atom_types)
    for atom_type in set_atom_types:
        idx: int = atom_types.index(atom_type) # the first index of the current atom_type in atom_types list
        Ls: List[int] = Ls_list[idx] # the Ls of the current atom_type
        Ls_dict[atom_type] = Ls
    return Ls_dict


def read_block_irreps_from_one_OpenMX_folder(folder: str, is_spin: bool) -> \
    Tuple[Dict[Tuple[int, int], Irreps], 
          Dict[Tuple[int, int], Irreps], 
          Dict[Tuple[int, int], List[int]], 
          Dict[Tuple[int, int], List[int]]]:
    """Get block irreps information from one OpenMX output folder."""
    block_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
    simple_sorted_block_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
    sort_dict: Dict[Tuple[int, int], List[int]] = {}
    inv_sort_dict: Dict[Tuple[int, int], List[int]] = {}
    
    # ======================从element.dat中读取元素类型信息，得到pair_types_unique======================
    # atom_types: 记录体系中各个原子的元素类型，即原子序数
    # atom_types_unique: 当前体系的元素类型，严格按照从小到大排序且不重复
    # pair_types_unique: 当前体系可能的原子对元素类型元组列表，元组(p_type, q_type)满足p_type<=q_type且不重复
    atom_types: List[int] = []  
    file: str = os.path.join(folder, 'element.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            atomic_number: int = int(line.split()[0])
            atom_types.append(atomic_number)
            line = f.readline()
    atom_types_unique: List[int] = list(set(atom_types))
    pair_types_unique: List[Tuple[int, int]] = []
    for i in range(len(atom_types_unique)):
        for j in range(i, len(atom_types_unique)):
            pair_types_unique.append(get_valid_pair_type(atom_types_unique[i], atom_types_unique[j]))
    # 只留下pair_types_unique和atom_types
    # (和block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict，is_spin)
    del file, f, line, atomic_number, atom_types_unique, i, j

    # ======================从type_orbitals.dat读取各原子的基函数角动量信息，得到Ls_list======================
    Ls_list: List[List[int]] = []
    file = os.path.join(folder, 'orbital_types.dat')
    with open(file) as f:
        line = f.readline()
        while line:
            Ls = [int(x) for x in line.split()] 
            # Ls: basis_group_Ls, angular momentum list for all basis group of the current atom
            Ls_list.append(Ls)
            line = f.readline()
    # 只留下Ls_list(和pair_types_unique, atom_types)
    # (和block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict，is_spin)
    del file, f, line, Ls
    
    # ======================计算block_dict的初始Irreps======================
    for pair_type in pair_types_unique:
        p_type, q_type = pair_type

        p: int = atom_types.index(p_type)
        q: int = atom_types.index(q_type)
        p_Ls: NDArray[np.int] = np.array(Ls_list[p])
        q_Ls: NDArray[np.int] = np.array(Ls_list[q])
        L_tuples: List[Tuple[int, int]] = []
        # 从p_Ls, q_Ls得到上一个阶段矩阵irreps中按照特定顺序排布的一维基函数组对分块矩阵的irreps中各个基函数组的角动量对L_tuples
        # 先p后q，顺序一定要保持和上面一样
        for p_single_L in p_Ls:
            for q_single_L in q_Ls:
                L_tuples.append(tuple([int(p_single_L), int(q_single_L)])) # 若不int()就是np.int64, 报错, out_js_list中的整数必须是python的int 
        # 只留下L_tuples, pair_type(和Ls_list, pair_types_unique, atom_types)
        # (和block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict，is_spin)
        del p_type, q_type, p, q, p_Ls, q_Ls, p_single_L, q_single_L

        block_irreps: Irreps = Irreps(None)
        for (l1, l2) in L_tuples:
            irreps_new = irreps_from_l1l2(l1, l2, mul=1, spinful=is_spin)[0]
            block_irreps = block_irreps + irreps_new 
        if is_spin:
            block_irreps = block_irreps + block_irreps
        block_irreps_dict[pair_type] = block_irreps
        # 只留下block_irreps, pair_type(和Ls_list, pair_types_unique, atom_types)
        # (和block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict，is_spin)
        del l1, l2, L_tuples, irreps_new
        
        #========================计算block_dict的Irreps的重排简化以及重排索引========================
        # 此时的block_irreps是零散的不利于e3nn--TensorProduct向量化处理，训练起来很慢
        # 因此把矩阵元重新排序，使得输入模型的simple_sorted_block_irreps是整块的，如"100x0e+200x1o+100x2e"
        # 为了保证矩阵元能够排序、复原，需要求出正排序、逆排序索引列表sort_index, inv_sort_index
        # 未排序 ---inv----> 已排序 ---p---> 未排序
        sorted_irreps, ps, invs = block_irreps.sort()
        simple_sorted_block_irreps_dict[pair_type] = sorted_irreps.simplify()
        # 构造sorted_irreps的索引列表，用于复原成block_irreps
        begin_end_list: NDArray = np.cumsum(
            np.array([0] + [irrep_group.dim for irrep_group in sorted_irreps])
            ) # len(begin_end_list) == len(ps) + 1 == max(ps) + 1 + 1
        ps: NDArray = np.array(ps) # tuple to ndarray 
        sort_dict[pair_type] = []
        for p in ps:
            sort_dict[pair_type] += list(
                range(begin_end_list[p], begin_end_list[p+1])
                )
        # 只留下block_irreps, invs(和Ls_list, pair_types_unique, atom_types)
        # (和block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict，is_spin)
        del begin_end_list, ps, p, sorted_irreps
        # 构造block_irreps的索引列表，用于排序成sort_irreps(或者说排序成simple_sort_irreps)
        begin_end_list: NDArray = np.cumsum(
            np.array([0] + [irrep_group.dim for irrep_group in block_irreps])
            ) # len(begin_end_list) == len(ps) + 1 == max(invs) + 1 + 1
        invs: NDArray = np.array(invs)
        inv_sort_dict[pair_type] = []
        for inv in invs:
            inv_sort_dict[pair_type] += list(
                range(begin_end_list[inv], begin_end_list[inv+1])
                )
        # (np.array(sort_dict[pair_type])[inv_sort_dict[pair_type]] == np.arange(0, block_irreps.dim)).all() == True
        

    return block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict


def read_block_irreps_from_OpenMX(folder_list: List[str], is_spin: bool) -> \
    Tuple[Dict[Tuple[int, int], Irreps], 
          Dict[Tuple[int, int], Irreps], 
          Dict[Tuple[int, int], List[int]], 
          Dict[Tuple[int, int], List[int]]]:
    """Get block irreps information from all OpenMX output folders,
    in case accidents such as: 
        only C-C pair type in the 1st-29th folder, 
        but C-C, C-H pair types in the 30th folder.
    If all systems contain the same pair type, this process is not needed."""
    block_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
    simple_sorted_block_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
    sort_dict: Dict[Tuple[int, int], List[int]] = {}
    inv_sort_dict: Dict[Tuple[int, int], List[int]] = {}

    for folder in folder_list:
        new_b, new_simple, new_sort, new_inv = \
            read_block_irreps_from_one_OpenMX_folder(folder=folder, is_spin=is_spin)
        block_irreps_dict.update(new_b)
        simple_sorted_block_irreps_dict.update(new_simple)
        sort_dict.update(new_sort)
        inv_sort_dict.update(new_inv)
    
    return block_irreps_dict, simple_sorted_block_irreps_dict, sort_dict, inv_sort_dict


def read_Ls_dict_from_OpenMX(folder_list: List[str]) -> Dict[int, List[int]]:
    """Get Ls_dict information from all OpenMX output folders.

    In case accidents such as:
    only C atom type in the 1st-29th folder, but C, H atom types in the 30th folder.
    If all systems contain the same atom type, this process is not needed."""
    Ls_dict: Dict[int, List[int]] = {}
    for folder in folder_list:
        Ls_dict.update(read_Ls_dict_from_one_OpenMX_folder(folder=folder))
    return Ls_dict


# = Basic Data Type for Graph Dataset =
def get_hetero_data(folder: str, is_spin: bool, default_dtype_torch: torch.dtype) -> HeteroData:
    data: HeteroData = HeteroData()
    node_fea_dict, edge_index_dict, edge_fea_dict= \
            read_x_dict_from_one_OpenMX_folder(folder=folder, is_spin=is_spin, default_dtype_torch=default_dtype_torch)
    edge_label_dict = \
            read_label_dict_from_one_OpenMX_folder(folder=folder, is_spin=is_spin, default_dtype_torch=default_dtype_torch)
    

    for node, node_fea in node_fea_dict.items():
        node: str = str(node) # node: atomic number
        data[node].x = node_fea


    for edge, edge_index in edge_index_dict.items():
        edge_fea = edge_fea_dict[edge]
        edge_label = edge_label_dict[edge]

        begin, end = edge # note that begin <= end
        begin = str(begin)
        end = str(end)
        edge: str = begin + "-" + end
        data[begin, edge, end].edge_index = edge_index
        data[begin, edge, end].edge_fea = edge_fea
        data[begin, edge, end].y = edge_label

    return data


import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData
# HeteroDataset不调用InMemoryDataset的初始化函数super.__init__
# 因为InMemoryDataset的初始化会调用self._process()，这里有很多语句我并不希望执行
# 详见pytorch-geometric.data.dataset中的Dataset类(InMemoryDataset类的父类)
class HeteroDataset(InMemoryDataset):
    # 覆盖掉Dataset中用property修饰的processed_dir方法
    # 避免报错AttributeError: property 'processed_dir' of 'HeteroDataset' object has no setter
    def processed_dir(self):
        return None
    
    def __init__(self, processed_dir: str, graph_file: str, is_spin: bool, default_dtype_torch: torch.dtype):
        """processed_dir: 经过预处理的数据的文件夹路径。
        graph_file: 图文件的存储路径。如果图文件存在即graph_file存在，那么就直接读取这个图文件."""
        # 不设置self._indices的话调用len()方法求数据集数量会报错
        self._indices = None
        # 不设置self.transform的话后续使用Dataloader调用__getitem__会报错
        # AttributeError: 'HeteroDataset' object has no attribute 'transform'
        self.transform = None

        # self.processed_dir覆盖掉原有的InMemoryDataset的property函数
        self.processed_dir: str = processed_dir 
        self.graph_file: str = graph_file
        self.is_graph_exist: bool = os.path.exists(graph_file)
        if self.is_graph_exist:
            logger.info(f"Graph file has already been generated.")
            logger.info(f"(Path of the Graph file: {graph_file})")
            logger.info(f"Read the existing graph file!")
            #TODO 当graph_file存在时，processed_dir不需要指定
        else:
            logger.info(f"Graph file has not been generated.")
            logger.info(f"(Path of the Graph file: {graph_file})")
            logger.info(f"Generate the graph file using data from processed_dir!")
            logger.info(f"(Path of the processed_dir={processed_dir})")
            # 当graph_file不存在时，processed_dir是一个必然存在的文件夹
            assert os.path.exists(processed_dir) and os.path.isdir(processed_dir), f"processed_dir={processed_dir} Does Not Exist!"
        self.is_spin: bool = is_spin
        self.default_dtype_torch: torch.dtype = default_dtype_torch

        self.Ls_dict: Dict[int, List[int]] = {}
        self.block_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
        self.simple_sorted_block_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
        self.sort_dict: Dict[Tuple[int, int], List[int]] = {}
        self.inv_sort_dict: Dict[Tuple[int, int], List[int]] = {}


        if self.is_graph_exist==False:
            folder_list = []
            logger.info(f'Looking for preprocessed data under: {self.processed_dir}')
            for root, dirs, files in os.walk(self.processed_dir):
                if {'element.dat', 'orbital_types.dat', 'lat.dat', 'site_positions.dat'}.issubset(files):
                    folder_list.append(root)
            assert len(folder_list) != 0, "Can not find any structure"
            # ========统计原子基函数轨道信息Ls_dict用于把矩阵从直和形式转化为张量积形式========
            self.Ls_dict = read_Ls_dict_from_OpenMX(folder_list=folder_list)

            # ========统计各原子对类型矩阵的Irreps用于初始化模型、给矩阵元排序和复原========
            self.block_irreps_dict, self.simple_sorted_block_irreps_dict, self.sort_dict, self.inv_sort_dict =\
                read_block_irreps_from_OpenMX(folder_list=folder_list, is_spin=self.is_spin)
            
            # ===================生成模型训练所需信息并存储为json文件===================
            # 记录self.block_irreps_dict, self.simple_sorted_block_irreps_dict, self.sort_dict, self.inv_sort_dict,
            # 和is_spin, default_dtype_torch 以及 Ls_dict
            # 确保读取已有pt文件,已有json文件后，可以直接训练不需要再输入这些参数
            bloc, simp, sor, inv_sor = {}, {}, {}, {} 
            for pair_type, v in self.block_irreps_dict.items():
                bloc[to_dtype_edge(pair_type, dtype=str)] = str(v) 
            for pair_type, v in self.simple_sorted_block_irreps_dict.items():
                simp[to_dtype_edge(pair_type, dtype=str)] = str(v) 
            for pair_type, v in self.sort_dict.items():
                sor[to_dtype_edge(pair_type, dtype=str)] = v
            for pair_type, v in self.inv_sort_dict.items():
                inv_sor[to_dtype_edge(pair_type, dtype=str)] = v

            info = dataset_info_recorder(
                is_spin=self.is_spin,
                default_dtype_torch=to_dtype_default_torch_dtype(self.default_dtype_torch, dtype=str),
                block_irreps_dict=bloc,
                simple_sorted_block_irreps_dict=simp,
                sort_dict=sor,
                inv_sort_dict=inv_sor,
                Ls_dict=self.Ls_dict
            )
            json_file: str = self.graph_file.replace(r".pt", r".json")
            with open(json_file, 'w') as f:
                json.dump(info._asdict(), f, indent=4) # Dict key int -> str, tuple -> raise Error

            del json_file, info
            del pair_type, v
            del bloc, simp, sor, inv_sor
            

            # ===================最关键的部分: 生成模型所需数据、给矩阵元重新排序并存储为pt文件===================
            data_list: List[HeteroData] = []

            for folder in tqdm.tqdm(folder_list):
                # 生成模型所需数据
                data: HeteroData = get_hetero_data(folder=folder, is_spin=self.is_spin, default_dtype_torch=self.default_dtype_torch)
                # 给重叠矩阵元排序，注意edge_fea[:, 0:4]=[:, distance, normed_vector]之后才是矩阵部分
                for edge, edge_fea in data.edge_fea_dict.items():
                    p_type, _, q_type = edge
                    p_type = int(p_type) 
                    q_type = int(q_type) 
                    pair_type: Tuple[int, int] = get_valid_pair_type(p_type, q_type)
                    data.edge_fea_dict[edge] = torch.cat(
                        [edge_fea[:, 0: 4], edge_fea[:, 4:][:, self.inv_sort_dict[pair_type]]], dim=-1
                    )
                '''
                # 给哈密顿矩阵元排序
                for edge, y in data.y_dict.items():
                    p_type, _, q_type = edge
                    p_type = int(p_type) 
                    q_type = int(q_type) 
                    pair_type: Tuple[int, int] = get_valid_pair_type(p_type, q_type)
                    data.y_dict[edge] = y[:, self.inv_sort_dict[pair_type]]
                '''
                # 把数据载入data_list
                data_list.append(data)
                    
            # 把List[Data]转化成graph_file这个路径对应的pt文件
            # 不要调用InMemoryDatase自带的self.save(data_list=data_list, path=self.graph_file)，它会把self.data转化为Dict，
            #       等到torch.load生成的pt文件时，调用from_dict把Dict变回Data会报错：TypeError: keywords must be strings
            torch.save(obj=self.collate(data_list), f=self.graph_file)
            # 节省内存
            del data_list, data
            del edge_fea, edge, p_type, q_type, pair_type
            
        # ===================读取已生成的pt文件===================
        # InMemoryDatase的self.data属性, self.slices属性: 调用torch.load函数便能自动初始化
        self.data: BaseData = None
        self.slices: Dict[str, Tensor] = None
        # 此时pt文件必然已经存在，读取pt文件，调用torch.load函数载入数据
        self.data, self.slices = torch.load(self.graph_file)

        # ===================读取已生成的json文件===================
        # *注意要和生成json部分的代码匹配
        # 此时json文件必然已经存在，读取json文件，加载self.is_spin，self.default_dtype_torch，
        # self.block_irreps_dict, self.simple_sorted_block_irreps_dict, self.sort_dict, self.inv_sort_dict
        json_file: str = self.graph_file.replace(r".pt", r".json")
        with open(json_file, 'r') as f:
            dct: Dict = json.load(f)
            info: dataset_info_recorder = dataset_info_recorder(**dct)

            self.is_spin: bool = info.is_spin
            # json.load donnot support torch.dtype, so string here
            self.default_dtype_torch: torch.dtype = to_dtype_default_torch_dtype(info.default_dtype_torch, dtype=torch.dtype)
            # json.load donnot support tuple as Dict key, so string here
            for pair_type, v in info.block_irreps_dict.items():
                self.block_irreps_dict[to_dtype_edge(pair_type, dtype=[Tuple, int])] = Irreps(v)
            for pair_type, v in info.simple_sorted_block_irreps_dict.items():
                self.simple_sorted_block_irreps_dict[to_dtype_edge(pair_type, dtype=[Tuple, int])] = Irreps(v)
            for pair_type, v in info.sort_dict.items():
                self.sort_dict[to_dtype_edge(pair_type, dtype=[Tuple, int])] = v
            for pair_type, v in info.inv_sort_dict.items():
                self.inv_sort_dict[to_dtype_edge(pair_type, dtype=[Tuple, int])] = v
            # json.load donnot support int as Dict key, so string here
            for atom_type, v in info.Ls_dict.items():
                self.Ls_dict[int(atom_type)] = v 
                
    def __repr__(self) -> str:
        s: str = f"GNN Dataset Information:\n"
        s += f"\tClass Name: {self.__class__.__name__}\n"
        s += f"\tMemory Usage: {self.statistic_memory()/(1024**3)} GB\n"
        s += f"\tNumber of Data (self.len()): {self.len()}\n"
        s += f"\tIs Spinful or Not (self.is_spin): {self.is_spin}\n"
        s += f"\tDtype of Data (self.default_dtype_torch): {self.default_dtype_torch}\n"

        s += f"\tGraphs Information: \n"
        num_nodes_list, num_S_edges_list, num_H_edges_list = self.statistic_graph()
        s += f"\t\tAverage number of nodes per graph: {sum(num_nodes_list)/len(num_nodes_list)}\n"
        s += f"\t\tAverage number of edges per graph (from S): {sum(num_S_edges_list)/len(num_S_edges_list)}\n"
        s += f"\t\tAverage number of edges per graph (from H): {sum(num_H_edges_list)/len(num_H_edges_list)}\n"

        s += f"\tBlocks Information: \n"
        for pair_type, irreps in self.simple_sorted_block_irreps_dict.items():
            s += f"\t\tPair Type {pair_type}: \n"
            s += f"\t\t\tBlock Size {irreps.dim}\n"
            s += f"\t\t\tBlock Irreps {irreps}\n"
        return s
    
    def statistic_memory(self) -> float:
        """Count memory in byte."""
        def get_tensor_bytes(tensor: Tensor) -> int:
            return tensor.numel() * tensor.element_size()
        
        memory: float = 0
        for idx in range(self.len()):
            data: HeteroData = self.get(idx=idx)
            node_types, edge_types = data.metadata()
            for node in node_types:
                memory += get_tensor_bytes(data[node].x)
            for edge in edge_types:
                memory += get_tensor_bytes(data[edge].edge_index)
                memory += get_tensor_bytes(data[edge].edge_fea)
                memory += get_tensor_bytes(data[edge].y)
        return memory
    
    def statistic_graph(self) -> Tuple[List[int], List[int], List[int]]:
        """Count total nodes number, edges number from S, edges number from H in each graph."""
        num_nodes_list: List[int] = []
        num_S_edges_list: List[int] = [] # from data[edge].edge_index, from overlaps
        num_H_edges_list: List[int] = [] # from data[edge].y, from hamiltonians
        for idx in range(self.len()):
            nodes_num: int = 0 # instead of num_nodes because easy typing
            S_edges_num: int = 0
            H_edges_num: int = 0

            data: HeteroData = self.get(idx=idx)
            node_types, edge_types = data.metadata()
            for node in node_types:
                nodes_num += len(data[node].x)
            for edge in edge_types:
                S_edges_num += data[edge].edge_index.shape[1] # shape[0] always equal to 2
                H_edges_num += data[edge].y.shape[0] # shape = (num_blocks, ...)

            num_nodes_list.append(nodes_num)
            num_S_edges_list.append(S_edges_num)
            num_H_edges_list.append(H_edges_num)
        return num_nodes_list, num_S_edges_list, num_H_edges_list

        
        






r"""
# 最终block按照simplified_irreps_dict排序，用于加速机器学习训练过程
            simplified_irreps_dict: Dict[Tuple[int, int], Irreps] = {}
            for edge, irreps in block_irreps_dict.items():
                sorted_irreps, _, inv = irreps.sort()
                simplified_irreps_dict[edge] = sorted_irreps.simplify()
                # suppose mul of all irrep in irreps == 1
                #TODO

"""
    

        