# Data Directory

Place datasets under `data/raw`.

MVTec AD example:

```text
data/raw/mvtec/
  bottle/
    train/good/*.png
    test/good/*.png
    test/broken_large/*.png
    ground_truth/broken_large/*_mask.png
```

VisA can be loaded from a split CSV or by recursive directory discovery. Keep large
dataset files out of version control.
