import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from utils.load_model import load_policy_from_checkpoint
from data import GC6DLIFT3DFormatDataset, collate_lift3d
from torch.utils.data import DataLoader
from models.graspnet_adapter import pred_decode_17d

def main():
    ckpt = os.path.join(ROOT, "checkpoints/gc6d_vggt_ft_adapter_graspnet_s4.pt")
    print("loading model...", flush=True)
    m = load_policy_from_checkpoint(
        ckpt,
        device="cuda",
        graspnet_ckpt="/home/ziyaochen/graspnet-baseline/logs/log_rs/checkpoint-rs.tar",
        graspnet_root="/home/ziyaochen/graspnet-baseline",
    )
    m.eval()
    print("model ok", type(m).__name__, flush=True)

    data_dir = "/mnt/ssd/ziyaochen/GraspClutter6D/offline_unified"
    camera = "realsense-d435"
    ds = GC6DLIFT3DFormatDataset(
        data_dir=data_dir,
        split="val",
        camera=camera,
        image_size=224,
        max_samples=1,
        load_gt_multi=True,
    )
    print("dataset len", len(ds), flush=True)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_lift3d, num_workers=0)
    batch = next(iter(loader))
    images, pcs, _, _, _, _, metas = batch
    images = images.cuda()
    pcs = pcs.cuda()
    with torch.no_grad():
        ep = m(pcs, images=images)
        out = pred_decode_17d(ep, torch.device("cuda"), max_grasps=256)
    print("forward ok", out.shape, flush=True)

if __name__ == "__main__":
    main()
