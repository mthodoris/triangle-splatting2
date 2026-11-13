"""Step 3 of notes/plan.md: the per-triangle texture must stay attached to its triangle.

Every triangle starts with a texture filled with its own ID (1, 2, ...) and Adam moments
filled with that ID too. After each operation that changes the triangle list
(prune_triangles, add_new_gs = split 1->4, _prune_vertices, final cleanup) we check:
  - number of textures == number of triangles (and the same for the Adam moments);
  - a texture with ID i sits next to the triangle that originally had ID i
    (same 3 vertex positions), and its Adam moments still say i;
  - new children have an all-zero texture and zero Adam moments (v1 split rule).
Also: save/load keeps the texture; texture_res=0 has no texture at all.

Usage (from combined/, venv active):  python tests/test_texture_sync.py
"""
import os
import sys
import tempfile
import types

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.triangle_model import TriangleModel, child_textures  # noqa: E402

dev = "cuda"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")


def make_model(texture_res, V=300, T=600, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    m = TriangleModel(3, texture_res)
    m.vertices = nn.Parameter(torch.rand(V, 3, device=dev, generator=g))
    tri = torch.stack([torch.randperm(V, device=dev, generator=g)[:3] for _ in range(T)])
    m._triangle_indices = torch.unique(torch.sort(tri, dim=1)[0], dim=0).to(torch.int32)
    T = m._triangle_indices.shape[0]
    m.vertex_weight = nn.Parameter(torch.randn(V, 1, device=dev, generator=g))
    m._features_dc = nn.Parameter(torch.randn(V, 1, 3, device=dev, generator=g))
    m._features_rest = nn.Parameter(torch.randn(V, 15, 3, device=dev, generator=g))
    m._sigma = 0.0
    m.image_size = torch.ones(T, device=dev)
    m.importance_score = torch.rand(T, device=dev, generator=g) + 0.1
    if texture_res > 0:
        m._texture = nn.Parameter(torch.zeros(T, 3, texture_res, texture_res, device=dev))
    args = types.SimpleNamespace(texture_lr=0.0025, split_texture="zero",
                                 position_lr_delay_mult=0.01, position_lr_max_steps=30000)
    m.training_setup(args, 0.0016, 0.03, 0.0015)
    m.add_percentage = 1.3
    m.size_probs_zero = 0.0
    m.size_probs_zero_image_space = 0.0
    return m


def adam_state(m):
    grp = m._texture_group()
    return m.optimizer.state.get(grp["params"][0], None)


def label(m):
    """Give every triangle an ID texture + ID Adam moments; return {id: vertex positions}."""
    # one Adam step with random gradients so that every group has Adam state
    for p in [m.vertices, m.vertex_weight, m._features_dc, m._features_rest, m._texture]:
        p.grad = torch.randn_like(p)
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    T = m._triangle_indices.shape[0]
    ids = torch.arange(1, T + 1, device=dev, dtype=torch.float).view(T, 1, 1, 1)
    with torch.no_grad():
        m._texture.copy_(ids.expand_as(m._texture))
    st = adam_state(m)
    st["exp_avg"].copy_(ids.expand_as(st["exp_avg"]))
    st["exp_avg_sq"].copy_(10 * ids.expand_as(st["exp_avg_sq"]))
    return {i + 1: positions(m, i) for i in range(T)}


def positions(m, k):
    p = m.vertices.detach()[m._triangle_indices[k].long()]       # [3, 3]
    return tuple(sorted(tuple(round(x, 6) for x in v.tolist()) for v in p))


def verify(m, original, step, expect_children=False):
    T = m._triangle_indices.shape[0]
    st = adam_state(m)
    check(f"{step}: #textures == #triangles", m._texture.shape[0] == T and st["exp_avg"].shape[0] == T
          and st["exp_avg_sq"].shape[0] == T, f"({T} triangles)")
    tex, ea, eas = m._texture.detach(), st["exp_avg"], st["exp_avg_sq"]
    ids = tex[:, 0, 0, 0]
    uniform = torch.all(tex == ids.view(-1, 1, 1, 1)).item()
    state_ok = torch.all(ea == ids.view(-1, 1, 1, 1)).item() and torch.all(eas == 10 * ids.view(-1, 1, 1, 1)).item()
    check(f"{step}: Adam moments moved with their texture", uniform and state_ok)
    wrong = sum(1 for k in range(T) if ids[k] > 0 and positions(m, k) != original[int(ids[k])])
    check(f"{step}: every texture still on its own triangle", wrong == 0, f"({wrong} misplaced)")
    n_children = int((ids == 0).sum())
    orig_sets = set(original.values())
    child_is_new = all(positions(m, k) not in orig_sets for k in range(T) if ids[k] == 0)
    if expect_children:
        check(f"{step}: children have zero texture + zero Adam state", n_children > 0 and child_is_new,
              f"({n_children} children)")
    else:
        check(f"{step}: no unexpected zero textures", n_children == 0)
    ids_kept = ids[ids > 0]
    check(f"{step}: no texture duplicated", ids_kept.unique().numel() == ids_kept.numel())


def main():
    print("child_textures")
    p = torch.randn(5, 3, 8, 8, device=dev)
    c = child_textures(p, "zero")
    check("zero mode: [4P, 3, S, S] zeros", c.shape == (20, 3, 8, 8) and torch.all(c == 0).item())
    try:
        child_textures(p, "copy")
        check("other modes not implemented yet -> error", False)
    except NotImplementedError:
        check("other modes not implemented yet -> error", True)

    print("\nprune_triangles")
    m = make_model(4)
    original = label(m)
    g = torch.Generator(device=dev).manual_seed(1)
    mask = torch.rand(m._triangle_indices.shape[0], device=dev, generator=g) > 0.3
    m.prune_triangles(mask)
    verify(m, original, "prune")

    print("\nadd_new_gs (split 1 -> 4, parents removed)")
    m = make_model(4)
    original = label(m)
    T0 = m._triangle_indices.shape[0]
    m.add_new_gs(iteration=500, cap_max=10**7, splitt_large_triangles=5)
    T1 = m._triangle_indices.shape[0]
    n_par = (T1 - T0) // 3          # each split: -1 parent, +4 children
    verify(m, original, "split", expect_children=True)
    check("split: #children == 4 x #parents", int((m._texture[:, 0, 0, 0] == 0).sum()) == 4 * n_par
          and (T1 - T0) % 3 == 0, f"({n_par} parents)")

    print("\n_prune_vertices (triangles that lose a vertex are dropped)")
    m = make_model(4)
    original = label(m)
    vmask = torch.rand(m.vertices.shape[0], device=dev, generator=g) > 0.1
    m._prune_vertices(vmask)
    verify(m, original, "vertex prune")

    print("\nseveral rounds of split / prune / vertex prune")
    m = make_model(4)
    original = label(m)
    for r in range(3):
        m.importance_score = torch.rand(m._triangle_indices.shape[0], device=dev, generator=g) + 0.1
        m.image_size = torch.ones(m._triangle_indices.shape[0], device=dev)
        m.add_new_gs(iteration=500, cap_max=10**7, splitt_large_triangles=5)
        m.prune_triangles(torch.rand(m._triangle_indices.shape[0], device=dev, generator=g) > 0.2)
        m._prune_vertices(torch.rand(m.vertices.shape[0], device=dev, generator=g) > 0.05)
    verify(m, original, "3 rounds", expect_children=True)

    print("\nsave / load")
    with tempfile.TemporaryDirectory() as d:
        m.save_parameters(d)
        m2 = TriangleModel(3)                 # render.py creates the model without texture_res
        m2.load_parameters(d)
        check("load: texture restored", m2.texture_res == 4 and torch.equal(m2._texture.detach(), m._texture.detach()))
        check("load: optimizer has a texture group", m2._texture_group() is not None)

    print("\ntexture_res = 0 (plain TS+)")
    m0 = make_model(0)
    check("no texture tensor, no texture group", m0._texture.numel() == 0 and m0._texture_group() is None)
    m0.add_new_gs(iteration=500, cap_max=10**7, splitt_large_triangles=5)
    m0.prune_triangles(torch.rand(m0._triangle_indices.shape[0], device=dev, generator=g) > 0.3)
    check("split + prune still work", m0._texture.numel() == 0)
    with tempfile.TemporaryDirectory() as d:
        m0.save_parameters(d)
        s = torch.load(os.path.join(d, "point_cloud_state_dict.pt"))
        check("saved file has no texture (same keys as plain TS+)", "texture" not in s and "texture_res" not in s)

    print("\nRESULT:", "PASS" if all(results) else "FAIL", f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
