# Sign Language (ASL) to Text Translation (Computer Vision)

![COMPUTER VISION — Static ASL alphabet classification](assets/portfolio-banner.svg)

Academic computer-vision project for translating static American Sign Language (ASL) alphabet gestures into text with a convolutional neural network and real-time OpenCV webcam inference. The model covers 29 classes: A–Z, `del`, `nothing`, and `space`.

## Published material

- `asl_to_text.ipynb` — project notebook for loading data, defining and training the CNN, evaluation, and webcam inference.
- `asl_inference.py` — small command-line entry point that loads the submitted model without training.
- `models/cnn_for_asl_grayscale.h5` — submitted 23.6 MiB trained model.
- [French academic report (PDF)](docs/academic-report-fr.pdf).

The external **ASL Alphabet** dataset is not redistributed because of its size. Download it from the [Kaggle dataset page](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

## Dataset

The project uses the public **ASL Alphabet Dataset** referenced in the academic report. It contains 29 classes: the 26 alphabet letters plus `del`, `nothing` and `space`.

Because the dataset is large, it is not redistributed in this repository. For training or evaluation, download the ASL Alphabet dataset from Kaggle (`grassknoted/asl-alphabet`) and place the training and test folders according to the notebook paths.

## Model contract and preprocessing

The CNN processes 200×200 grayscale images and produces probabilities across the 29 ASL classes. The project preprocessing converts images to grayscale, resizes them to the model input dimensions and reshapes them to a single channel.

The output order is:

```text
A, B, C, D, del, E, F, G, H, I, J, K, L, M, N,
nothing, O, P, Q, R, S, space, T, U, V, W, X, Y, Z
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Use the provided model

From the repository root, run a single-image prediction:

```bash
python asl_inference.py --image path/to/a/200x200-or-larger-image.jpg
```

Or open `asl_to_text.ipynb` and run the import, configuration, helper-function, model-loading, and desired inference cells. The provided trained model is stored at `models/cnn_for_asl_grayscale.h5`.

## Optional evaluation

Place the official test images as follows:

```text
asl_alphabet_test/asl_alphabet_test/*.jpg
```

In the notebook, run the setup/helper cells, load the provided model, then uncomment and execute the optional evaluation cell. Evaluation is separate from both inference and training.

## Optional training

Training uses the public ASL Alphabet training folders organized by class. The notebook contains the CNN definition and training workflow described in the academic report.

## Webcam inference

```bash
python asl_inference.py --webcam
```

Place a static sign inside the central 200×200 region and press `q` to exit. The academic report presents real-time webcam examples for the alphabet and the three special classes.

## Results and limitations

The project report evaluates the CNN on the ASL test set and reports:

- **Accuracy:** `92.86%`
- **Loss:** `11.27`
- **29 output classes:** the alphabet plus `del`, `nothing` and `space`
- Real-time webcam recognition through OpenCV

The qualitative tests presented in the report show real-time predictions for the alphabet and the three special classes.

The limitations identified in the report are the quality and diversity of the images, confusion between visually similar gestures, additional processing time on less powerful machines, and the absence of data augmentation. Proposed improvements include data augmentation, deeper or pre-trained models, extension toward multiple hands and complete phrases, and further optimization of real-time inference.

## Authors

- Adam El Akkaoui
- Mohammed Zaidouh
