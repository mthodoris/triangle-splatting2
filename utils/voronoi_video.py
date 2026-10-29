#
# Join the Voronoi progress frames (<model>/voronoi/frames/frame_<iter>.jpg, written by train.py with
# --vor_frames_interval) into an MP4 and a smaller GIF. train.py calls this at the end of training; it can also
# be run by hand on a run folder:
#   python utils/voronoi_video.py <model_path> [--fps 10] [--gif_width 960]
#

import glob
import os
from argparse import ArgumentParser

import cv2
from PIL import Image


def make_video(frames_dir, out_mp4, out_gif=None, fps=10, gif_width=960):
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not files:
        return 0
    first = cv2.imread(files[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    gif = []
    for f in files:
        img = cv2.imread(f)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)
        if out_gif:
            small = cv2.resize(img, (gif_width, round(h * gif_width / w)), interpolation=cv2.INTER_AREA)
            gif.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).quantize(256, dither=Image.Dither.NONE))
    writer.release()
    if gif:
        gif[0].save(out_gif, save_all=True, append_images=gif[1:], duration=int(1000 / fps), loop=0)
    return len(files)


if __name__ == "__main__":
    p = ArgumentParser(description="Voronoi progress frames -> MP4 + GIF")
    p.add_argument("model_path")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--gif_width", type=int, default=960)
    a = p.parse_args()
    d = os.path.join(a.model_path, "voronoi")
    n = make_video(os.path.join(d, "frames"), os.path.join(d, "progress.mp4"), os.path.join(d, "progress.gif"), a.fps, a.gif_width)
    print("{} frames -> {}".format(n, os.path.join(d, "progress.mp4 / progress.gif")))
