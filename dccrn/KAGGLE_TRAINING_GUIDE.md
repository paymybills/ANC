# Kaggle Training Guide: DCCRN

Since training DCCRN on an RTX 2050 (4GB) is tight, **Kaggle's P100 (16GB)** is the perfect upgrade. Here is how to migrate this codebase to a Kaggle Notebook.

## 1. Setup the Notebook
Create a new Python Notebook on Kaggle and enable **GPU P100** in the Settings.

## 2. Structure your Code
In the first cell, copy-paste the core model and utility classes. You can combine them into one cell or use Kaggle's "Utility Script" feature.

### File Migration Mapping:
1.  **Helper Functions**: Copy `dccrn/complex_nn.py`, `dccrn/dccrn_model.py`, and `dccrn/audio_utils.py` into one big code cell titled "Model Definition".
2.  **Dataset**: Copy `dccrn/dataset.py` into a cell titled "Data Processing".
3.  **Loss**: Copy `dccrn/losses.py` into a "Loss Functions" cell.
4.  **Training**: Use the `train.py` logic in your final execution cell.

## 3. Data Source
Search and add these datasets to your Kaggle environment:
*   **Speech**: Search for "DNS Challenge" or "LibriSpeech".
*   **Noise**: Search for "MUSAN" or "Environmental Noise".

Update the paths in your training script:
```python
clean_dir = "/kaggle/input/librispeech-clean/train"
noise_dir = "/kaggle/input/musan-noise/musan"
```

## 4. Execution Logic
In the final cell, run the training loop:

```python
# Example execution cell in Kaggle
args = argparse.Namespace(
    clean_dir=clean_dir,
    noise_dir=noise_dir,
    save_dir="/kaggle/working/checkpoints",
    epochs=100,
    batch_size=32, # Kaggle P100 can handle 32 easily
    lr=1e-3,
    segment_length=2.0, # Use longer segments for better learning
    num_workers=4
)

train(args)
```

## 5. Pro Tips for Kaggle
*   **Persistence**: Kaggle notebooks session out after 12 hours. Ensure `save_dir` points to `/kaggle/working/` and download your `.pt` files periodically.
*   **Batch Size**: If you get "Out of Memory" (OOM), reduce `batch_size` to 16.
*   **Internet**: Keep Internet "On" in settings if you need to download specific libraries.
