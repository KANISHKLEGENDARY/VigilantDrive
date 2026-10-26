"""Check train/validation image splits for exact hashes and likely shared filename sources."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
import yaml


# Resolve a split entry from dataset YAML to its image directory.
def image_folder(root, value):
    path = Path(value)
    folder = path if path.is_absolute() else root / path
    return folder if folder.name.lower() == "images" else folder / "images"


# Return the source-like filename prefix before a Roboflow hash marker.
def filename_prefix(path):
    stem = path.stem.lower()
    return stem.split(".rf.", 1)[0]


# Hash each image in chunks so large datasets do not need to fit in memory.
def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Calculate a perceptual hash that can flag visually similar resized or recompressed images.
def perceptual_hash(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8].flatten()[1:]
    median_value = float(np.median(low_frequency))
    value = 0
    for bit in low_frequency:
        value = (value << 1) | int(bit > median_value)
    return value


# Compare train and validation images and print potential leakage clearly.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/Driver_Drowsiness_YOLO.training_ready/data.yaml")
    parser.add_argument("--config", default="config.yaml", help="Config file supplies the near-duplicate hash threshold")
    args = parser.parse_args()
    yaml_path = Path(args.data).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root_value = Path(config.get("path", "."))
    root = root_value if root_value.is_absolute() else yaml_path.parent / root_value
    train_dir, valid_dir = image_folder(root, config["train"]), image_folder(root, config["val"])
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise FileNotFoundError(f"Train or validation image folder is missing: {train_dir}, {valid_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train, valid = [p for p in train_dir.rglob("*") if p.suffix.lower() in extensions], [p for p in valid_dir.rglob("*") if p.suffix.lower() in extensions]
    train_hashes = {file_hash(p): p.name for p in train}
    valid_hashes = {file_hash(p): p.name for p in valid}
    duplicate_hashes = sorted(set(train_hashes) & set(valid_hashes))
    runtime_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    phash_limit = int(runtime_config.get("near_duplicate_hash_max_distance", 4))
    train_visual = [(path, perceptual_hash(path)) for path in train]
    valid_visual = [(path, perceptual_hash(path)) for path in valid]
    near_duplicates = []
    for train_path, train_hash in train_visual:
        if train_hash is None:
            continue
        for valid_path, valid_hash in valid_visual:
            if valid_hash is not None and (train_hash ^ valid_hash).bit_count() <= phash_limit:
                near_duplicates.append((train_path.name, valid_path.name))
    train_prefixes = {filename_prefix(p): p.name for p in train}
    valid_prefixes = {filename_prefix(p): p.name for p in valid}
    shared_prefixes = sorted(prefix for prefix in set(train_prefixes) & set(valid_prefixes) if prefix)
    problems = bool(duplicate_hashes or shared_prefixes or near_duplicates)
    print(f"Train images: {len(train)}; validation images: {len(valid)}")
    print((f"PROBLEM: {len(duplicate_hashes)} exact image hashes appear in both splits"
           if duplicate_hashes else "OK: no exact image hashes are shared"))
    print((f"PROBLEM: {len(shared_prefixes)} possible shared source filename prefixes found"
           if shared_prefixes else "OK: no source-like prefixes are shared"))
    print(f"PROBLEM: {len(near_duplicates)} possible near-duplicate image pairs found (perceptual-hash distance <= {phash_limit})"
          if near_duplicates else f"OK: no near-duplicates found at configured distance {phash_limit}")
    if duplicate_hashes:
        for value in duplicate_hashes[:10]:
            print(f"  identical: train={train_hashes[value]} validation={valid_hashes[value]}")
    if shared_prefixes:
        for prefix in shared_prefixes[:10]:
            print(f"  prefix {prefix}: train={train_prefixes[prefix]} validation={valid_prefixes[prefix]}")
    for train_name, valid_name in near_duplicates[:10]:
        print(f"  similar images: train={train_name} validation={valid_name}")
    print("Prefix matching is a filename heuristic; driver/video identity cannot be confirmed unless the names encode it.")
    if not problems:
        print("OK: no overlap detected by the two available checks.")


if __name__ == "__main__":
    main()
