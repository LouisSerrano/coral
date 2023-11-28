# Copyright 2022 Yuan Yin & Matthieu Kirchmeyer

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torchdiffeq import odeint

def scheduling(_int, _f, true_codes, t, epsilon, method="rk4"):
    if epsilon < 1e-3:
        epsilon = 0
    if epsilon == 0:
        codes = _int(_f, y0=true_codes[0], t=t, method=method)
    else:
        eval_points = np.random.random(len(t)) < epsilon
        eval_points[-1] = False
        eval_points = eval_points[1:]

        start_i, end_i = 0, None
        codes = []
        for i, eval_point in enumerate(eval_points):
            if eval_point == True:
                end_i = i + 1
                t_seg = t[start_i : end_i + 1]
                res_seg = _int(_f, y0=true_codes[start_i], t=t_seg, method=method)

                if len(codes) == 0:
                    codes.append(res_seg)
                else:
                    codes.append(res_seg[1:])
                start_i = end_i
        t_seg = t[start_i:]
        res_seg = _int(_f, y0=true_codes[start_i], t=t_seg, method=method)
        if len(codes) == 0:
            codes.append(res_seg)
        else:
            codes.append(res_seg[1:])
        codes = torch.cat(codes, dim=0)
    return codes

def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf