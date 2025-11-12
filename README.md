# Bach Chorale Generator Using Recurrent Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Overview](#overview)
- [Author](#author)
- [Deep Learning Approach](#deep-learning-approach)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Implementation Details](#implementation-details)
  - [Model Architecture](#model-architecture)
  - [Data Processing](#data-processing)
  - [Training Process](#training-process)
  - [Generation Algorithm](#generation-algorithm)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Overview
This repository contains an implementation of a Bach chorale generator using Long Short-Term Memory (LSTM) networks. The model learns from the JSB Chorales dataset to understand the intricate patterns of four-part harmony in Bach's compositions, then generates new musical pieces that maintain the stylistic characteristics of baroque chorales. The system includes audio synthesis capabilities to convert the generated note sequences into playable audio.

## Author
- Sashwat Desai (desai.sas@northeastern.edu)
  - MS, Applied Mathematics
  - Northeastern University, Boston

## Deep Learning Approach
The implementation uses an LSTM-based neural network to model sequential patterns in Bach chorales:
- **LSTM Architecture**: 64-unit LSTM layer to capture temporal dependencies in musical sequences
- **Four-Part Harmony**: Simultaneous prediction of all four voices (Soprano, Alto, Tenor, Bass)
- **Sequence Modeling**: Uses 16-timestep sequences for training and generation
- **Normalization**: Min-max scaling of MIDI note values for stable training
- **Autoregressive Generation**: Iterative prediction of next timesteps based on previous context

## Key Features
- Complete end-to-end implementation from data processing to audio generation
- Audio synthesis using sine wave generation for playback
- Pre-trained model on JSB Chorales dataset
- Support for four-part harmony generation
- CSV export of generated chorales
- Real-time audio playback capabilities using IPython Audio

## Repository Structure
- **code/** - Implementation code
  - **Bach Chorale Generator.ipynb** - Main Jupyter notebook containing the full implementation
  - **placeholder.md** - Placeholder file
- **data/** - JSB Chorales dataset (train/test/validation splits)
  - **train/** - Training chorales in CSV format
  - **test/** - Test chorales in CSV format
  - **validation/** - Validation chorales in CSV format
  - **placeholder.md** - Placeholder file
- **model/** - Saved model weights
  - **model_Bach.keras** - Pre-trained LSTM model weights
  - **placeholder.md** - Placeholder file
- **output/** - Generated chorales
  - **generated_chorale.csv** - Sample generated chorale output
  - **placeholder.md** - Placeholder file
- **LICENSE** - MIT License
- **README.md** - Project documentation

## Implementation Details

### Model Architecture
The neural network architecture consists of:
- **Input Shape**: (16, 4) - sequences of 16 timesteps, each with 4 notes
- **LSTM Layer**: 64 units with return_sequences=True for sequence-to-sequence learning
- **Output Layer**: Dense layer with 4 units and sigmoid activation
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001
- **Metrics**: Mean Absolute Error (MAE)

### Data Processing
The data pipeline includes:
1. **Dataset**: JSB Chorales dataset containing Bach's four-part chorales
2. **Sequence Creation**: Sliding windows of length 16 to create input-output pairs
3. **Normalization**: Min-max scaling of MIDI note values to [0, 1] range
4. **Data Structure**: Each chorale represented as (timesteps × 4 voices)
5. **Train/Valid/Test Split**: Pre-divided dataset for proper evaluation

### Training Process
The model training involves:
- Initial training on single chorale for concept validation
- Full training on complete dataset with validation monitoring
- ModelCheckpoint callback to save best model based on validation loss
- EarlyStopping with patience=10 to prevent overfitting
- Batch size of 64 for efficient training
- 50 epochs maximum with early stopping

### Generation Algorithm
The chorale generation process:
1. **Seed Selection**: Random selection of 16-timestep seed sequence from test set
2. **Iterative Prediction**: Generate 100 new timesteps autoregressively
3. **Clipping**: Ensure predictions stay within [0, 1] range
4. **Denormalization**: Convert back to MIDI note range
5. **Rounding**: Convert to integer MIDI note values
6. **Audio Synthesis**: Convert notes to audio using sine wave generation at specific frequencies

## Results
- **Test Loss**: ~0.0089 (MSE)
- **Test MAE**: ~0.0677
- Generated chorales exhibit Bach-like harmonic progressions

### Audio Generation
The implementation includes a custom audio synthesis function that:
- Maps MIDI notes to frequencies using a piano note reference table
- Generates sine waves at corresponding frequencies
- Combines all four voices for polyphonic playback
- Uses 22050 Hz sampling rate for audio quality

## Setup and Installation
```bash
# Clone this repository
git clone https://github.com/desai-sashwat/bach-chorale-generator.git
cd bach-chorale-generator

# Install required packages
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install jupyter>=1.0.0
pip install ipython>=7.0.0
```

### Dataset Setup
The JSB Chorales dataset should be organized as:
```
jsb_chorales/
├── train/     # Training chorales (*.csv files)
├── valid/     # Validation chorales (*.csv files)
└── test/      # Test chorales (*.csv files)
```

## Usage
The main implementation is in the Jupyter notebook `code/Bach Chorale Generator.ipynb`. 

### Quick Start
1. Open the notebook:
```bash
jupyter notebook "code/Bach Chorale Generator.ipynb"
```

2. Run all cells to:
   - Load and explore the audio generation utilities
   - Process the JSB Chorales dataset
   - Train the LSTM model (or load pre-trained weights)
   - Generate new chorales
   - Convert to audio and playback

### Using Pre-trained Model
```python
# Load the saved model
from tensorflow.keras.models import load_model
model_Bach = load_model('model/model_Bach.keras')

# Generate a new chorale
chorale = generate_chorale(model_Bach, X_test_norm)

# Convert to audio and play
Audio(gen_data_0 + gen_data_1 + gen_data_2 + gen_data_3, rate=22050)
```

## Future Work
Potential enhancements for this project include:
- Implementing deeper LSTM architectures or bidirectional LSTMs
- Adding attention mechanisms for better long-range dependencies
- Incorporating music theory constraints as additional loss terms
- Supporting variable-length generation
- Adding tempo and dynamics variation
- Implementing MIDI export functionality
- Creating a web interface for real-time generation and playback
- Exploring transformer architectures for music generation
- Adding style transfer capabilities for different composers

## License
This project is licensed under the MIT License - see the LICENSE file for details.
