#!/usr/bin/env python3
"""
convert_to_classification.py

Converts a YOLO detection dataset (images + labels/*.txt) into an
ImageNet-style classification dataset with **named** class folders,
using a data.yaml (or classes.txt) mapping if provided.

Behavior:
- Reads labels/*.txt (YOLO detection lines: "<class_id> x y w h")
- If a classes mapping is provided (data.yaml or classes.txt), uses names
  for output folders (e.g. train/acanthaluteres_vittiger/...). Otherwise
  numeric class ids are used ("0", "1", ...).
- Skips images with no labels; skips images with multiple labels by default.
- Optionally --force-first to take the first label when multiple exist.
- Automatically creates train/val split per class with --val-split.
"""

import argparse
import os
import shutil
import random
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def find_image_for_label(label_path, images_dir):
    base = os.path.splitext(os.path.basename(label_path))[0]
    # direct extensions check
    for ext in IMG_EXTS:
        cand = os.path.join(images_dir, base + ext)
        if os.path.isfile(cand):
            return cand
    # recursive search (useful if images are in subfolders)
    for root, _, files in os.walk(images_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            if name == base and ext.lower() in IMG_EXTS:
                return os.path.join(root, f)
    return None

def parse_label_file(label_path):
    class_ids = []
    with open(label_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_ids.append(parts[0])
    return class_ids

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_image_to(output_dir, split, class_name, image_path, move=False):
    dest_dir = os.path.join(output_dir, split, class_name)
    ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, os.path.basename(image_path))
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(os.path.basename(image_path))
        i = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(dest_dir, f"{name}_{i}{ext}")
            i += 1
    if move:
        shutil.move(image_path, dest_path)
    else:
        shutil.copy2(image_path, dest_path)
    return dest_path

def load_classes_from_txt(path):
    """
    classes.txt : each line is a class name in index order (0-based)
    """
    names = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if s:
                names.append(s)
    return {str(i): names[i] for i in range(len(names))}

def load_classes_from_yaml(path):
    """
    Attempts to read data.yaml. Supports:
    1) names: [ 'a', 'b', ... ]   (list)
    2) names:
         0: class_a
         1: class_b
       or
       names:
         - class_a
         - class_b
    Falls back to basic parsing if PyYAML not installed.
    Returns dict mapping string class_id -> class_name.
    """
    try:
        import yaml
    except Exception:
        yaml = None

    if yaml is not None:
        with open(path, 'r', encoding='utf-8') as fh:
            parsed = yaml.safe_load(fh)
        if not parsed:
            return {}
        names = parsed.get('names', None)
        if isinstance(names, dict):
            # keys may be int or strings
            return {str(k): v for k, v in names.items()}
        elif isinstance(names, list):
            return {str(i): names[i] for i in range(len(names))}
        else:
            # unexpected format
            return {}
    else:
        # simple fallback parser (line-based). Not full YAML compliant, but handles common YOLO data.yaml styles.
        mapping = {}
        names_started = False
        list_mode = False
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                raw = line.rstrip('\n')
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('names:'):
                    # could be "names: [a, b, c]" or "names:" followed by lines
                    rest = line[len('names:'):].strip()
                    if rest.startswith('[') and rest.endswith(']'):
                        # inline list: names: [a, b, c]
                        inner = rest[1:-1]
                        items = [i.strip().strip('\'"') for i in inner.split(',') if i.strip()]
                        mapping = {str(i): items[i] for i in range(len(items))}
                        return mapping
                    else:
                        # enter block parsing mode
                        names_started = True
                        # determine if block is mapping (starts with digit+:) or list (starts with '-' lines)
                        continue
                if names_started:
                    if line.startswith('-'):
                        list_mode = True
                        val = line[1:].strip().strip('\'"')
                        idx = len(mapping)
                        mapping[str(idx)] = val
                    else:
                        # try key: value
                        if ':' in line:
                            key, val = line.split(':', 1)
                            key = key.strip()
                            val = val.strip().strip('\'"')
                            if key != '':
                                mapping[str(key)] = val
                        else:
                            # unknown line, ignore
                            pass
        return mapping

def load_class_mapping(classes_path):
    if not classes_path:
        return {}
    if not os.path.isfile(classes_path):
        print(f"[WARN] classes file '{classes_path}' not found. Falling back to numeric IDs.")
        return {}
    ext = os.path.splitext(classes_path)[1].lower()
    if ext in ('.txt',):
        try:
            return load_classes_from_txt(classes_path)
        except Exception as e:
            print(f"[WARN] failed to parse classes.txt: {e}. Falling back to numeric IDs.")
            return {}
    elif ext in ('.yaml', '.yml'):
        try:
            mapping = load_classes_from_yaml(classes_path)
            if mapping:
                return mapping
            else:
                print("[WARN] parsed data.yaml but 'names' not found or empty. Falling back to numeric IDs.")
                return {}
        except Exception as e:
            print(f"[WARN] failed to parse data.yaml: {e}. Falling back to numeric IDs.")
            return {}
    else:
        print(f"[WARN] unknown classes file extension '{ext}'. Supported: .txt, .yaml. Falling back to numeric IDs.")
        return {}

def main(args):
    random.seed(args.seed)

    class_map = load_class_mapping(args.classes)
    if class_map:
        print(f"Using class mapping from {args.classes} with {len(class_map)} entries.")
    else:
        print("No class mapping found — numeric class IDs will be used as folder names.")

    label_files = []
    for root, _, files in os.walk(args.labels_dir):
        for f in files:
            if f.lower().endswith('.txt'):
                label_files.append(os.path.join(root, f))

    print(f"Found {len(label_files)} label files in {args.labels_dir}")

    samples = []
    skipped = 0
    multi = 0
    missing_image = 0

    for lf in label_files:
        class_ids = parse_label_file(lf)
        if not class_ids:
            skipped += 1
            continue

        if len(class_ids) > 1:
            multi += 1
            if args.force_first:
                chosen = class_ids[0]
            else:
                # skip images with multiple labels by default
                continue
        else:
            chosen = class_ids[0]

        img = find_image_for_label(lf, args.images_dir)
        if img is None:
            missing_image += 1
            continue

        # map to name if available
        class_name = class_map.get(str(chosen), None)
        if class_name is None:
            # fallback to numeric id (string)
            class_name = str(chosen)
        # sanitize: replace spaces with underscore
        class_name = class_name.replace(' ', '_')
        samples.append((img, class_name))

    print(f"Usable samples: {len(samples)} (skipped empty: {skipped}, skipped multi: {multi if not args.force_first else 0}, missing images: {missing_image})")

    # group by class
    by_class = defaultdict(list)
    for img, cname in samples:
        by_class[cname].append(img)

    # create output structure and split per class
    total_copied = 0
    for cname, imgs in by_class.items():
        random.shuffle(imgs)
        n_val = int(len(imgs) * args.val_split)
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for im in train_imgs:
            copy_image_to(args.output_dir, 'train', cname, im, move=args.move)
            total_copied += 1
        for im in val_imgs:
            copy_image_to(args.output_dir, 'val', cname, im, move=args.move)
            total_copied += 1

    # print distribution
    print("\nClass distribution in output (total, train, val):")
    for cname, imgs in by_class.items():
        n_total = len(imgs)
        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val
        print(f"  {cname}: {n_total}, train {n_train}, val {n_val}")

    print(f"\nTotal files copied/moved: {total_copied}")
    print(f"Output root: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Convert YOLO detection labels -> ImageNet-style classification folders (named classes)")
    p.add_argument('--images-dir', required=True, help="Path to images folder")
    p.add_argument('--labels-dir', required=True, help="Path to labels folder (YOLO .txt files)")
    p.add_argument('--output-dir', required=True, help="Output root folder (will contain train/val subfolders)")
    p.add_argument('--classes', required=False, help="Optional path to data.yaml or classes.txt to map numeric IDs to names")
    p.add_argument('--val-split', type=float, default=0.2, help="Fraction to reserve for validation (0.0 - 0.5 recommended)")
    p.add_argument('--move', action='store_true', help="Move files instead of copying")
    p.add_argument('--force-first', action='store_true',
                   help="If a label file has multiple class lines, force-assign the first class instead of skipping (use with care)")
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(args)
