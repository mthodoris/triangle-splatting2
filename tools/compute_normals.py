"""Precompute Metric3D normal maps exactly like TS+ does, one image per process.

Why: TS+ computes missing normal maps inside `readColmapCameras`
(scene/dataset_readers.py), all in one process. On a 12 GB GPU, Metric3D ViT-large
at `images_2` resolution runs out of memory from the second image on
(notes/decisions.md, 2026-11-13). This script does the same computation for each
image in a fresh Python process, so memory starts clean every time.

Same as TS+:
  - model:  torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True), fp32, eval
  - input:  PIL RGB -> ToTensor -> resize_to_multiple(28) (imported from TS+ itself)
  - output: prediction_normal[:, :3] -> bilinear back to image size -> (n+1)/2 -> uint8 PNG (BGR via cv2)
  - names/order: COLMAP sparse/0/images.bin, `<images>/<name>` -> `<normals>/<stem>.png`,
    with `<normals>` = images folder name with "images" replaced by "normals"

All-or-nothing: PNGs go to `<normals>.partial/` first (re-running resumes there).
Only when every image succeeds is that folder renamed to `<normals>/`. TS+ then
just reads them. An existing `<normals>/` is never touched.

Usage (from combined/, venv active):
  python tools/compute_normals.py -s /data/manousis/360_v2/bonsai -i images_2
"""
import argparse
import os
import subprocess
import sys
import time

COMBINED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMBINED)


def image_names(source):
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
    try:
        extr = read_extrinsics_binary(os.path.join(source, "sparse/0", "images.bin"))
    except Exception:
        extr = read_extrinsics_text(os.path.join(source, "sparse/0", "images.txt"))
    return [os.path.basename(extr[k].name) for k in extr]  # same order as TS+


def compute_one(image_path, out_path):
    """Body copied from scene/dataset_readers.py:readColmapCameras (the `else` branch)."""
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from scene.dataset_readers import resize_to_multiple

    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
    model.eval().cuda()

    image = Image.open(image_path)
    image_extract = image.convert("RGB")
    original_size = image_extract.size[::-1]  # (H, W)
    rgb = transforms.Compose([transforms.ToTensor()])(image_extract).unsqueeze(0).cuda()
    with torch.no_grad():
        rgb_resized = resize_to_multiple(rgb, multiple=28)
        pred_depth, _, output_dict = model.inference({'input': rgb_resized})
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        pred_normal_resized = torch.nn.functional.interpolate(
            pred_normal, size=original_size, mode='bilinear', align_corners=False)
    normals = pred_normal_resized.squeeze().detach().cpu().numpy()
    normals = np.transpose(normals, (1, 2, 0))
    normals_vis = ((normals + 1.0) / 2.0).clip(0, 1)
    normals_uint8 = (normals_vis * 255).astype(np.uint8)
    tmp = out_path + ".tmp.png"
    if not cv2.imwrite(tmp, cv2.cvtColor(normals_uint8, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cv2.imwrite failed for {tmp}")
    os.replace(tmp, out_path)  # never leave a half-written PNG under the final name
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", help="scene folder (with sparse/0 and the images folder)")
    ap.add_argument("-i", "--images", default="images")
    ap.add_argument("--retries", type=int, default=1, help="extra attempts per image")
    ap.add_argument("--single", nargs=2, metavar=("IMAGE", "OUT_PNG"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:  # worker mode: one image, then exit
        compute_one(*args.single)
        return

    images_dir = os.path.join(args.source, args.images)
    final_dir = images_dir.replace("images", "normals")  # same rule as TS+
    partial_dir = final_dir + ".partial"
    if os.path.exists(final_dir):
        sys.exit(f"{final_dir} already exists; not touching it.")
    os.makedirs(partial_dir, exist_ok=True)

    names = image_names(args.source)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    failed = []
    t0 = time.time()
    for n, name in enumerate(names, 1):
        out = os.path.join(partial_dir, os.path.splitext(name)[0] + ".png")
        if os.path.exists(out):
            continue
        for attempt in range(1 + args.retries):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--single",
                                os.path.join(images_dir, name), out],
                               env=env, cwd=COMBINED, capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(out):
                peak = [l for l in r.stdout.splitlines() if l.startswith("peak")]
                print(f"[{n}/{len(names)}] {name} ok ({peak[-1] if peak else ''}, "
                      f"{time.time() - t0:.0f}s total)", flush=True)
                break
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            print(f"[{n}/{len(names)}] {name} attempt {attempt + 1} FAILED: {err}", flush=True)
        else:
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} image(s) failed: {failed}\nPartial results kept in {partial_dir} "
              f"(re-run to resume). {final_dir} was NOT created.")
        sys.exit(1)
    os.rename(partial_dir, final_dir)
    print(f"\nAll {len(names)} normal maps done -> {final_dir}")


if __name__ == "__main__":
    main()
