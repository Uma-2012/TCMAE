# TC-MAE: Unsupervised Video Anomaly Detection using Lightweight Temporal Context Masked Autoencoder

This repository implements an Unsupervised Video Anomaly Detection system. The  model utilizes a Masked Autoencoder (MAE) architecture with a Convolutional  Vision Transformer (CvT) backbone.

1. INTRODUCTION
--------------------------------------------------------------------------------
The system is designed to learn normality from normal video data by masking random patches of the input and learning to reconstruct the original pixel values. During inference, the model fails to reconstruct unseen abnormal events accurately, resulting in a high reconstruction error that serves as the anomaly score.

2. KEY FEATURES
--------------------------------------------------------------------------------
* **Input Representation:** Takes a temporal stack of 3 RGB frames as input (Previous, Current, Next), resulting in a 9-channel tensor [3x3]. **Backbone:** Uses a Convolutional Vision Transformer (CvT) which introduces convolutional inductive biases into the Vision Transformer architecture. **Masking Strategy:** Randomly masks a high percentage (e.g., 50-75%) of input patches to force the model to learn semantic context.**Anomaly Scoring:** Uses Mean Squared Error (MSE) between the reconstructed frame and the ground truth frame. **Post-Processing:** Applies Gaussian temporal smoothing to frame-level scores to reduce noise.

3. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
.
├── configs/
│   └── configs.py              # Configuration for Avenue and ShanghaiTech datasets
├── data/
│   ├── train_dataset.py        # Loads 3-frame stacks for training (Normal data)
│   └── test_dataset.py         # Loads 3-frame stacks + Labels for testing
├── model/
│   ├── cvt.py                  # Convolutional Vision Transformer backbone
│   ├── mae_cvt.py              # MAE architecture & Reconstruction logic
│   └── model_factory.py        # Helper functions to build model variants
├── util/
│   ├── abnormal_utils.py       # Gaussian smoothing filters
│   ├── misc.py                 # Logging and distributed training utils
│   ├── morphology.py           # Differentiable morphological ops (optional)
│   └── time_benchmark.py       # Inference speed benchmarking
├── engine_train.py             # Training and Validation epoch loops
├── extract_gradients.py        # Preprocessing script (Frames & Gradients)
├── inference.py                # Final evaluation and AUC calculation
└── main.py                     # Entry point for training and testing

4. PREREQUISITES
--------------------------------------------------------------------------------
Ensure you have the following Python libraries installed:

* Python 3.8+
* PyTorch (with CUDA support)
* numpy
* opencv-python
* timm
* einops
* scikit-learn
* ml_collections
* tensorboard

5. DATASET PREPARATION
--------------------------------------------------------------------------------
1.  Download the **CUHK Avenue**, **ShanghaiTech**, **UCSD Ped 1 & Ped 2** datasets.
2. CONFIGURATION: define the correct path in config.py
--------------------------------------------------------------------------------
All hyperparameters are managed in `configs/configs.py`. 

Key parameters to check before running:
* `dataset_path`: Path to your dataset.
* `output_dir`: Where checkpoints and logs will be saved.
* `batch_size`: Adjust according to your GPU memory (Default: 18).
* `mask_ratio`: Percentage of patches to mask (Default: 0.5 or 0.75).
* `run_type`: Set to 'train' for training or 'inference' for evaluation.

7. USAGE
--------------------------------------------------------------------------------

[Training]
To train the model on the Avenue dataset:
    $ python3 main.py --dataset avenue

To train on ShanghaiTech:
    $ python3 main.py --dataset shanghai
    
To train on UCSD_Ped1:
    $ python3 main.py --dataset ucsd_ped1

To train on UCSD_Ped2:
    $ python3 main.py --dataset ucsd_ped2

[Inference / Resume]
To resume training or run inference, modify the `run_type` in `configs.py` or use the resume flag:
    $ python3 main.py --dataset avenue --resume /path/to/checkpoint.pth
