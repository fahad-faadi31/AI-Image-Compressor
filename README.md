#  AI Image Compression

A deep learning–based image compression system built with **PyTorch**. The project learns compact latent representations of images using a convolutional autoencoder with residual connections and quantization, enabling efficient compression while preserving visual quality.

>  This project is under active development. The current focus is building a high-quality neural image codec before integrating a web/mobile application.

---

#  Features

- Deep convolutional autoencoder
- Residual encoder and decoder architecture
- Learned latent representation
- Straight-Through Estimator (STE) quantization
- Hybrid Loss (MSE + SSIM)
- PSNR and SSIM evaluation
- Image compression & decompression
- JPEG comparison utilities
- TensorBoard logging
- Automatic checkpoint saving
- Visualization after every few epochs
- Cosine Annealing Learning Rate Scheduler
- Automatic Mixed Precision (AMP) support
- Modular project architecture

---

#  Project Structure

```
AI-Image-Compression/
│
├── src/
│   ├── blocks.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── model.py
│   ├── dataset.py
│   ├── loss.py
│   ├── metrics.py
│   ├── quantization.py
│   └── ...
│
├── train.py
├── evaluate.py
├── compress.py
├── decompress.py
├── compare.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

#  Model Architecture

```
Input Image
      │
      ▼
Encoder
      │
Residual Blocks
      │
Quantization (STE)
      │
Latent Representation
      │
Residual Decoder
      ▼
Reconstructed Image
```

---

#  Training Pipeline

The model is trained using:

- Dataset: DIV2K
- Patch Size: 128×128
- Optimizer: AdamW
- Learning Rate Scheduler: Cosine Annealing
- Mixed Precision Training (AMP)
- Gradient Clipping
- Automatic Checkpoint Saving

---

#  Loss Function

The model uses a hybrid objective:

```
Loss = α × MSE + β × (1 − SSIM)
```

where:

- **MSE** preserves pixel accuracy.
- **SSIM** improves structural similarity and perceptual quality.

---

#  Evaluation Metrics

The following metrics are tracked during training:

- Validation Loss
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

TensorBoard logs are automatically generated.

---

#  Training

```
python train.py --epochs 150
```

Example:

```
python train.py --epochs 150 --batch_size 32
```

---

#  Compress an Image

```
python compress.py
```

---

#  Decompress an Image

```
python decompress.py
```

---

#  Evaluate Model

```
python evaluate.py
```

---

#  Compare with JPEG

```
python compare.py
```

---

#  Dataset

The project uses the **DIV2K** high-resolution image dataset.

Training images are extracted into patches for efficient learning.

Dataset is **not included** in this repository due to size.

---

#  Technologies Used

- Python
- PyTorch
- TorchVision
- PIL (Pillow)
- TensorBoard
- NumPy
- Matplotlib
- tqdm

---

#  Current Progress

 Data Pipeline

 Residual Blocks

 Encoder

 Decoder

 Quantization

 Hybrid Loss

 Training Pipeline

 Compression

 Decompression

 Evaluation

 JPEG Comparison

 Long GPU Training

 Model Optimization

 Entropy Coding

 FastAPI Backend

 Flutter/Web Application

---

#  Future Improvements

- Entropy Coding
- ONNX Export
- Faster Inference
- Support for PNG, JPEG, JPG, BMP, WebP, TIFF and HEIC
- Custom `.aic` compressed file format
- FastAPI REST API
- Flutter/Web Client
- Docker Deployment

---

#  Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository and open a pull request.

---

#  License

This project is released under the MIT License.

---

#  Author

**Muhammad Fahad**

Software Engineering Student  
University of Engineering and Technology (UET) Lahore

GitHub:
https://github.com/fahad-faadi31
