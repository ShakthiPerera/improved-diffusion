import os
import argparse
import glob
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch_fidelity
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from prdc import compute_prdc

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity")

MAX_IMAGES = 50000

# -------- Clean a subset of files (non-images, corrupted) --------
def clean_image_files(file_paths):
    """
    Given a list of file paths, filter out non-image and corrupted files.
    At most len(file_paths) images are opened.
    """
    valid_exts = [".jpg", ".jpeg", ".png"]
    cleaned = []

    for f in tqdm(file_paths, desc="Cleaning subset"):
        ext = os.path.splitext(f)[1].lower()
        if ext not in valid_exts:
            print(f"Removing non-image file: {f}")
            try:
                os.remove(f)
            except OSError:
                pass
            continue
        try:
            img = Image.open(f)
            img = img.convert("RGB")  # enforce RGB
            img.verify()             # check corruption
            cleaned.append(f)
        except Exception:
            print(f"Removing corrupted file: {f}")
            try:
                os.remove(f)
            except OSError:
                pass
    return cleaned

# -------- Prepare resized copy with torchvision transforms --------
def prepare_resized_copy(file_paths, size=64):
    """
    Create a temporary directory with resized/cropped copies of given file paths.
    Only processes the provided list.
    """
    tmp_dir = tempfile.mkdtemp()
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])
    for f in tqdm(file_paths, desc="Processing images for FID/IS"):
        try:
            img = Image.open(f).convert("RGB")
            img = transform(img)
            img.save(os.path.join(tmp_dir, os.path.basename(f)))
        except Exception:
            print(f"Skipping invalid file during resize: {f}")
    return tmp_dir

# -------- Dataset for PRDC --------
class ImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

# -------- Extract raw RGB images as numpy arrays for PRDC --------
def extract_rgb_features(dataloader):
    imgs_list = []
    for imgs, _ in tqdm(dataloader, desc="Collecting RGB images for PRDC"):
        # imgs: float tensor [B,C,H,W] in [0,1]
        imgs_uint8 = (imgs * 255).to(torch.uint8)
        imgs_np = imgs_uint8.permute(0, 2, 3, 1).cpu().numpy()  # B,H,W,C uint8
        imgs_list.append(imgs_np)
    return np.concatenate(imgs_list, axis=0)

# -------- Main --------
def main(real_dir, gen_dir, image_size=64, batch_size=32, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # List all files (no loading yet)
    real_all = sorted(glob.glob(os.path.join(real_dir, "*")))
    gen_all = sorted(glob.glob(os.path.join(gen_dir, "*")))

    if len(real_all) == 0 or len(gen_all) == 0:
        raise RuntimeError("No files found in real_dir or gen_dir.")

    # We will only ever open at most MAX_IMAGES from each folder
    max_n = min(len(real_all), len(gen_all), MAX_IMAGES)

    real_subset = real_all[:max_n]
    gen_subset = gen_all[:max_n]

    print(f"[Info] Candidate images (before cleaning): real={len(real_subset)}, gen={len(gen_subset)}, cap={MAX_IMAGES}")

    print("[Cleaning subsets (non-image / corrupted files)]")
    real_clean = clean_image_files(real_subset)
    gen_clean = clean_image_files(gen_subset)

    if len(real_clean) == 0 or len(gen_clean) == 0:
        raise RuntimeError("No valid images left in real_dir or gen_dir after cleaning subset.")

    # Now ensure equal count between cleaned subsets
    n = min(len(real_clean), len(gen_clean))
    real_files = real_clean[:n]
    gen_files = gen_clean[:n]

    print(f"[Info] Using {n} images from real_dir and gen_dir (max {MAX_IMAGES}).")

    print("[Preparing resized copies for FID/IS]")
    real_tmp = prepare_resized_copy(real_files, size=image_size)
    gen_tmp = prepare_resized_copy(gen_files, size=image_size)

    print("[Calculating FID and IS with torch-fidelity]")
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_tmp,
        input2=real_tmp,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        verbose=False
    )
    fid_score = metrics['frechet_inception_distance']
    is_mean = metrics['inception_score_mean']
    is_std = metrics['inception_score_std']

    shutil.rmtree(real_tmp)
    shutil.rmtree(gen_tmp)

    print("[Calculating PRDC metrics with raw RGB images]")
    prdc_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # uint8 [0,255]; we scale in extract_rgb_features   #############
    ])
    ds_real = ImageDataset(real_files, prdc_transform)
    ds_gen = ImageDataset(gen_files, prdc_transform)
    dl_real = DataLoader(ds_real, batch_size=batch_size, shuffle=False)
    dl_gen = DataLoader(ds_gen, batch_size=batch_size, shuffle=False)

    real_feats = extract_rgb_features(dl_real)
    gen_feats = extract_rgb_features(dl_gen)

    real_feats = real_feats.reshape(real_feats.shape[0], -1)
    gen_feats = gen_feats.reshape(gen_feats.shape[0], -1)
    prdc_metrics = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=5)

    print("\n==== Final Metrics ====")
    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    print(f"Precision: {prdc_metrics['precision']:.4f}")
    print(f"Recall: {prdc_metrics['recall']:.4f}")
    print(f"Density: {prdc_metrics['density']:.4f}")
    print(f"Coverage: {prdc_metrics['coverage']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID, IS, PRDC metrics on image directories")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images")
    parser.add_argument("--image_size", type=int, default=32, help="Image resolution (eg. 32 or 64)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for PRDC feature extraction")
    args = parser.parse_args()

    main(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )