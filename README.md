# Sign Language (ASL) to Text Translation (Computer Vision)

![COMPUTER VISION — ASL alphabet classification](assets/portfolio-banner.svg)


Academic computer-vision project for translating static American Sign Language (ASL) alphabet gestures into text with a convolutional neural network and real-time OpenCV webcam inference. The model covers 29 classes: A–Z, `del`, `nothing`, and `space`.

## Published material

- `asl_to_text.ipynb` — project notebook for loading data, defining and training the CNN, evaluation, and webcam inference.
- `models/cnn_for_asl_grayscale.h5` — trained CNN model used in the project.
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

## Project environment

The project uses Python with TensorFlow/Keras and OpenCV, as documented in the report and notebook.

## Use the provided model

Open `asl_to_text.ipynb` and follow the project cells for model loading, evaluation and webcam inference. The trained model is stored at `models/cnn_for_asl_grayscale.h5`.

## Optional evaluation

Place the official test images as follows:

```text
asl_alphabet_test/asl_alphabet_test/*.jpg
```

In the notebook, run the setup/helper cells, load the provided model, then uncomment and execute the optional evaluation cell. Evaluation is separate from both inference and training.

## Optional training

Training uses the public ASL Alphabet training folders organized by class. The notebook contains the CNN definition and training workflow described in the academic report.

## Webcam inference

The notebook contains the OpenCV real-time pipeline used in the project. A central region of interest is captured from the webcam, converted to grayscale, prepared for the CNN, and the predicted class is displayed on screen. The academic report presents real-time examples for the alphabet and the three special classes.

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
