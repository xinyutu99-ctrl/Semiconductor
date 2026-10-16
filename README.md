# WaferScope

A static, browser-side semiconductor image classification experiment. It trains a small TensorFlow.js CNN and reports metrics on a held-out test split.

## Run locally

Open a terminal in this folder and run:

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000>. TensorFlow.js is loaded from jsDelivr, so an internet connection is required to train. The images and model inputs are processed in the browser and are not uploaded to an application server.

## Prepare a dataset

Choose the `data` folder from the downloaded Carinthia dataset. WaferScope reads the semicolon-delimited `carinthia.csv`, matches each `file_name` to a JPEG in `images/`, and uses a binary rule: one chosen class is normal and the other classes are defective. Class 3 is selected by default; check the original Carinthia label legend before relying on the binary interpretation. The dataset has no wafer/sample ID column, so this dataset is split by image and cannot establish performance on unseen wafers.

The CSV has 4,591 labeled images with a strongly imbalanced class distribution. To keep browser memory manageable and avoid training on a normal-only prefix, WaferScope deterministically selects an equal number from the chosen normal class and the combined remaining classes (up to 600 per side). The displayed metrics describe this balanced image subset; precision and accuracy do not represent the original dataset prevalence.

You can also use a folder with labeled class folders and one separate folder per wafer/sample:

```text
dataset/
  normal/
    wafer-001/
      image-01.jpg
      image-02.jpg
    wafer-002/
      image-01.jpg
  defective/
    wafer-101/
      image-01.jpg
    wafer-102/
      image-01.jpg
```

Use `normal` and `defective` as the class folder names. The page requires at least four distinct sample folders in each class and 16 images total. Sample IDs must identify the physical wafer/sample consistently across both class folders; a shared ID is kept wholly in one split even when it has images with both labels. A fixed seed (42917) makes the assignment reproducible. This folder format is limited to 1,200 images per browser run.

The CNN resizes images to 64 × 64 RGB, scales pixel values to 0–1, and augments training images with random flips and quarter-turn rotations. The test split is not used during training. Metrics are computed at a 0.5 prediction threshold. The model and dataset are kept only for the current page session; reloading clears them.

## Deploy to Vercel

Import this folder as a project in Vercel and use the static site defaults: no framework preset, no build command, and the project root as the output directory. The included `index.html` is the entry point. The site is static; no login, API route, environment variable, or backend is required.

## Limits

A small or unrepresentative dataset cannot establish performance on production wafers. For datasets without wafer/sample IDs, image-level splits may put visually related images on both sides and should not be described as wafer-level validation. Keep sample IDs accurate and do not change the held-out set to chase a target score.
