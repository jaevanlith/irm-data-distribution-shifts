# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F


class DataModel(object):
    def __init__(self, dim, shift, dim_x=0, dim_z=0, ones=True, scramble=False, hetero=1, confounder_on_x=False):
        self.hetero = hetero

        # Possibility for different dimensions for X and Z
        if dim_x == 0 and dim_z == 0:
            self.dim_x = dim // 2
            self.dim_z = dim // 2
        elif dim_z > dim_x and dim_z + dim_x == dim:
            self.dim_x = dim_x
            self.dim_z = dim_z
        else:
            raise ValueError("Dimensions of z should be greater than dimensions of x! "
                             "AND total dimensions should add up!")

        if ones:
            # Invariant relationship from X to Y and Y to Z set to Identity
            self.wxy = torch.eye(self.dim_x)
            # Pad identity matrix with zeros
            source = torch.eye(self.dim_x)
            self.wyz = F.pad(input=source, pad=(0, (self.dim_z - self.dim_x), 0, 0), mode='constant', value=0)
        else:
            # Invariant relationship from X to Y  and Y to Z set to Gaussian
            self.wxy = torch.randn(self.dim_x, self.dim_x) / dim + 1
            self.wyz = torch.randn(self.dim_x, self.dim_z) / dim + 1

        if scramble:
            self.scramble, _ = torch.linalg.qr(torch.randn(dim, dim))
        else:
            self.scramble = torch.eye(dim)

        if confounder_on_x:
            # X related to confounder
            self.whx = torch.eye(self.dim_x)
        else:
            # Confounder never relates to X
            self.whx = torch.zeros(self.dim_x, self.dim_x)

        if shift == "CS":
            # No relations
            self.why = torch.zeros(self.dim_x, self.dim_x)
            self.whz = torch.zeros(self.dim_x, self.dim_z)
            self.wyz = torch.zeros(self.dim_x, self.dim_z)
        elif shift == "CF":
            # Confounder relates to Y and Z
            self.why = torch.randn(self.dim_x, self.dim_x) / dim
            self.whz = torch.randn(self.dim_x, self.dim_z) / dim
            self.wyz = torch.zeros(self.dim_x, self.dim_z)

            if confounder_on_x:
                self.whx = torch.randn(self.dim_x, self.dim_x) / dim

        elif shift == "AC":
            # Y relates to Z
            self.why = torch.zeros(self.dim_x, self.dim_x)
            self.whz = torch.zeros(self.dim_x, self.dim_z)
        elif shift == "HB":
            # Confounder relates to Y and Z
            # Y relates to Z
            self.why = torch.randn(self.dim_x, self.dim_x) / dim
            self.whz = torch.randn(self.dim_x, self.dim_z) / dim

            if confounder_on_x:
                self.whx = torch.randn(self.dim_x, self.dim_x) / dim

        else:
          raise ValueError("Shift should be CS, CF, AC or HB!")

    def solution(self):
        w = torch.cat((self.wxy.sum(1), torch.zeros(self.dim_z))).view(-1, 1)
        return w, self.scramble.t()

    def __call__(self, n, env):
        h = torch.randn(n, self.dim_x) * env
        x = h @ self.whx + torch.randn(n, self.dim_x) * env

        if self.hetero == 1:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim_x) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim_z)
        elif self.hetero == 0:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim_x)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim_z) * env
        else:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim_x)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim_z)

        return torch.cat((x, z), 1) @ self.scramble, y.sum(1, keepdim=True)

    def instantiate_training_environments(self, n, env):
        h = torch.randn(n, self.dim_x) * env
        x = h @ self.whx + torch.randn(n, self.dim_x) * env

        if self.hetero == 1:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim_x) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim_z)
        else:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim_x)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim_z) * env

        return x, y, z
