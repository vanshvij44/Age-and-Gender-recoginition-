Age and Gender Prediction
Overview

Predicts age group and gender from face images using a ResNet50 backbone and a Squeeze-and-Excitation (SE) block built with the Keras Functional API. The project is implemented in a Jupyter/Colab notebook age-gender-revised.ipynb.

Key points (brief)

Backbone: ResNet50 (include_top=False) pretrained on ImageNet.

Architecture: Keras Functional API with an SE (Squeeze-and-Excitation) block applied to feature maps, followed by task-specific heads (age-group classification and gender classification).

Outputs: one softmax for age-group classes and one softmax/binary output for gender.

Notebook-ready: designed to run in Google Colab.

Requirements

Python 3.8+

TensorFlow (2.6+) / Keras

numpy, pandas, scikit-learn

opencv-python, pillow

matplotlib, tqdm
Install:

pip install tensorflow numpy pandas scikit-learn opencv-python pillow matplotlib tqdm

Dataset & Input format

Image folder or CSV mapping: each row or sample should include image_path, age (or age_group), gender.

Recommended preprocessing: detect & crop face, resize to 224x224, scale pixel values to [0,1] or apply the ResNet50 preprocessing utility.

Model design (concise)

Input: Input(shape=(224,224,3)).

Backbone: ResNet50(include_top=False, weights='imagenet', input_tensor=inputs).

SE block (example behavior): global pooling → dense bottleneck → scale feature maps → multiply with backbone output.

Pooling: GlobalAveragePooling2D() after SE.

Shared dense layer(s) (optional) then two heads:

Age head: Dense(num_age_groups, activation='softmax')

Gender head: Dense(2, activation='softmax' or 'sigmoid')

Compile with appropriate losses:

Age: categorical_crossentropy

Gender: binary_crossentropy or categorical_crossentropy (depending on encoding)

Use weighted sum or multiple losses via a dictionary when compiling.



Training & evaluation

Use ImageDataGenerator or tf.data pipelines with augmentation (flip, brightness, small shift).

Use class weighting or oversampling if classes are imbalanced.

Recommended callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping.

Evaluate both tasks separately (age-group accuracy/confusion matrix and gender accuracy).








