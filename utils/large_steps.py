#
# Smooth ("large steps") updates for the mesh vertices, after Nicolet et al.,
# "Large Steps in Inverse Rendering of Geometry", SIGGRAPH Asia 2021.
#
# The mesh vertices x are optimized through u = (I + lambda L) x, with L the
# uniform (combinatorial) Laplacian of the mesh. The gradient w.r.t. u is
# (I + lambda L)^-1 dL/dx, which spreads each vertex's gradient over its
# neighbourhood, and u is updated with AdamUniform (Adam with one shared
# second-moment normalization, so the update stays smooth). Vertices then
# follow x = (I + lambda L)^-1 u: they move as a smooth surface instead of one
# by one, so local folds and self-intersections cannot form.
#
# The systems are solved with Jacobi-preconditioned conjugate gradients on the
# GPU, warm-started from the previous solution. The operator is built for one
# mesh topology: rebuild it after the mesh is refined. The mesh vertices are
# addressed by their indices in the model (idx), in that order.
#

import torch


class LargeSteps:
    def __init__(self, vertices, faces, lam, cg_iters=10, x_iters=5, betas=(0.9, 0.999), eps=1e-8):
        """vertices: [V,3] initial mesh vertex positions; faces: [F,3] mesh faces (indices < V)."""
        f = faces.long()
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        e = torch.unique(torch.sort(e, dim=1).values, dim=0)
        self.i, self.j = e[:, 0], e[:, 1]
        self.num_vertices = vertices.shape[0]
        ones = torch.ones(self.i.shape[0], device=vertices.device)
        self.deg = torch.zeros(self.num_vertices, device=vertices.device).index_add_(0, self.i, ones).index_add_(0, self.j, ones)[:, None]
        self.lam = lam
        self.diag = 1.0 + lam * self.deg  # Jacobi preconditioner
        self.cg_iters = cg_iters  # gradient solve: each step's gradient is new, so warm starts help less
        self.x_iters = x_iters  # position solve: u changes little per step, warm starts converge fast
        self.b1, self.b2 = betas
        self.eps = eps

        x = vertices.detach().clone()
        self.u = self.matvec(x)
        self.x = x
        self.g_u = torch.zeros_like(x)
        self.m1 = torch.zeros_like(x)
        self.m2 = torch.zeros_like(x)
        self.step_count = 0

    def matvec(self, x):
        """(I + lambda L) x with L = D - A."""
        ax = torch.zeros_like(x).index_add_(0, self.i, x[self.j]).index_add_(0, self.j, x[self.i])
        return x + self.lam * (self.deg * x - ax)

    def solve(self, b, x0, iters):
        """(I + lambda L)^-1 b by preconditioned CG from x0; returns the solution and the relative residual."""
        x = x0.clone()
        r = b - self.matvec(x)
        z = r / self.diag
        p = z.clone()
        rz = (r * z).sum(0)  # one CG per coordinate column
        bnorm = b.norm() + 1e-30
        for _ in range(iters):
            ap = self.matvec(p)
            alpha = rz / ((p * ap).sum(0) + 1e-30)
            x += alpha * p
            r -= alpha * ap
            z = r / self.diag
            rz_new = (r * z).sum(0)
            p = z + (rz_new / (rz + 1e-30)) * p
            rz = rz_new
        return x, (r.norm() / bnorm).item()

    @torch.no_grad()
    def step(self, vertices, lr, idx=None):
        """Update the mesh rows idx (default: the first num_vertices) of vertices (an nn.Parameter with .grad)
        and zero their gradient."""
        n = self.num_vertices
        if idx is None:
            idx = torch.arange(n, device=vertices.device)
        assert idx.numel() == n
        if vertices.grad is None:
            # densification/pruning replaced the parameter after backward: no gradient this step
            return float("nan")
        grad = vertices.grad[idx]
        self.g_u, _ = self.solve(grad, self.g_u, self.cg_iters)
        self.step_count += 1
        self.m1.mul_(self.b1).add_(self.g_u, alpha=1 - self.b1)
        self.m2.mul_(self.b2).add_(self.g_u.square(), alpha=1 - self.b2)
        m1 = self.m1 / (1 - self.b1 ** self.step_count)
        m2 = self.m2 / (1 - self.b2 ** self.step_count)
        self.u -= lr * m1 / (self.eps + m2.sqrt().max())
        self.x, residual = self.solve(self.u, self.x, self.x_iters)
        vertices.data[idx] = self.x
        vertices.grad[idx] = 0  # the regular optimizer must not move them too
        return residual
