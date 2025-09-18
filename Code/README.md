# ArtiFinger: Artificial Fingerprint Recognition System for Qin Terracotta Warriors


## Key Features

- Support for multiple backbone network architectures
- High-precision fingerprint feature extraction
- LFW dataset evaluation support
- Flexible training and prediction interfaces

## Project Structure

```
ArtiFinger/
├── arcface.py          # ArcFace model wrapper class
├── train.py            # Training script
├── eval_LFW.py         # LFW dataset evaluation script
├── predict.py          # Prediction script
├── nets/               # Network model definitions
│   ├── __init__.py
│   ├── arcface.py      # ArcFace network structure
│   └── artifingerNet.py # Custom fingerprint recognition network
└── utils/              # Utility functions
    ├── __init__.py
    ├── callback.py     # Callback functions
    ├── dataloader.py   # Data loading utilities
    ├── losses.py       # Loss functions
    ├── utils.py        # General utility functions
    ├── utils_fit.py    # Training utility functions
    └── utils_metrics.py # Evaluation metric functions
```