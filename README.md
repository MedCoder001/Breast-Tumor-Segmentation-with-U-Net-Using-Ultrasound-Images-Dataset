# Breast Tumor Segmentation with U-Net Using Ultrasound Images Dataset

This project focuses on segmenting breast tumor regions from ultrasound images using U-Net, a convolutional neural network architecture widely used for biomedical image segmentation tasks. This technology underlies many modern image generation models, such as DALL-E, Midjourney, and Stable Diffusion.

## Dataset

The dataset used in this project consists of breast ultrasound images along with corresponding masks that delineate benign and malignant tumor regions. The dataset is organized into three classes: benign, malignant, and normal. The dataset can be found on Kaggle and you can download it using the following link:

[Breast Ultrasound Images Dataset](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)

## Approaches
- Data was loaded and preprocessed by resizing, normalizing, and padding to make them suitable for model training.

- To the model building, I implemented U-Net architecture using TensorFlow and Keras, defined the model's backbone (e.g., ResNet34) and then compiled the model with custom loss functions (Dice loss and Binary Focal loss) and evaluation metrics (IOU and F-Score).

- The U-Net model was trained on the ultrasound images and corresponding masks iterating over epochs, loading batches of data, and updating model weights using the Adam optimizer.

- The trained model was evaluated on test data to assess its segmentation performance. I visualized sample images alongside their ground truth masks and predicted masks.

- The model weights was then saved for future use or deployment.

## Note 

1. Ensure you have the necessary libraries installed (`numpy`, `pandas`, `opencv-python`, `scipy`, `matplotlib`, `tensorflow`, `keras`, `segmentation-models`).
2. Mount Google Drive and set the Kaggle configuration directory if using Colab.
3. Download the breast ultrasound images dataset from Kaggle.
4. Run the provided Python script to execute the breast tumor segmentation system.
5. Modify the code as needed, such as changing the backbone architecture or adjusting hyperparameters.
6. Visualize the model's segmentation results and interpret the performance metrics.

## Conclusion

This project demonstrates the use of U-Net for segmenting breast tumor regions from ultrasound images, providing valuable insights for medical diagnosis and treatment planning. By accurately identifying tumor boundaries, healthcare professionals can make informed decisions and improve patient outcomes.

