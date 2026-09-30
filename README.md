# Sign Language (ASL) to Text Translation (Computer Vision)

Academic computer-vision project for classifying static American Sign Language alphabet images with a convolutional neural network and displaying predictions from a webcam region of interest.

## Verified contents

The cleaned notebook defines dataset loaders, grayscale preprocessing, a three-convolution-layer Keras model, batched training, evaluation, label decoding, and an OpenCV webcam loop. It covers 29 labels: A–Z plus `del`, `nothing`, and `space`.

The submitted notebook contains recorded evidence of training over 86,851 image files in batches and an evaluation on 28 test images. Its recorded result is accuracy `0.9285714` with loss `11.2689457`. These are historical notebook outputs, not an independently reproduced benchmark; outputs were removed from the public notebook because they included machine-specific paths and extensive logs.

## Technologies

Python, TensorFlow/Keras, OpenCV, NumPy, and h5py.

## Installation and use

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Place the external ASL Alphabet dataset in the paths expected by the notebook:

```text
asl_alphabet_train/asl_alphabet_train/<label>/*.jpg
asl_alphabet_test/asl_alphabet_test/*.jpg
```

Execute the notebook to train and save `cnn_for_asl_grayscale.h5`, then run its webcam cells. Press `q` to leave the webcam loop.

## Excluded material

The 1.1 GB dataset, the submitted 24.7 MB trained model, reports with embedded captures, and local execution outputs are not included. The dataset and model provenance/licensing were not established in the local files strongly enough for redistribution.

## Limitations

The system classifies isolated static signs rather than continuous sign language. It does not translate sentences, model motion, or provide linguistic interpretation. Webcam performance depends on framing and lighting, and the recorded test set is very small.

## Authors

- Adam El Akkaoui
- Mohammed Zaidouh
