---
title: "Vanilla Neural Network"
excerpt: "Comparison of different DNN architectures on multiple challenging maps on LG's SVL simualor"
header:
  overlay_filter: 0.6
  overlay_image: https://media.giphy.com/media/16quBBr5bnK8kdOAVy/giphy.gif
  teaser: https://media.giphy.com/media/16quBBr5bnK8kdOAVy/giphy.gif
  caption: "Photo credit: [**3blue1brown**](https://www.3blue1brown.com/)"
sidebar:
  - title: "Role"
    # url: [Youtube](/www.youtube.com/)
    image: https://media.giphy.com/media/16quBBr5bnK8kdOAVy/giphy.gif
    image_alt: "logo"
    text: ""
  - title: "Language & libraries:"
    text: "Python3 + OpenCV + Sklearn"
gallery:
  - url: /assets/images/unsplash-gallery-image-1.jpg
    image_path: assets/images/unsplash-gallery-image-1-th.jpg
    alt: "placeholder image 1"
  - url: /assets/images/unsplash-gallery-image-2.jpg
    image_path: assets/images/unsplash-gallery-image-2-th.jpg
    alt: "placeholder image 2"
  - url: /assets/images/unsplash-gallery-image-3.jpg
    image_path: assets/images/unsplash-gallery-image-3-th.jpg
    alt: "placeholder image 3"
toc: true
toc_label: "Jump to"
toc_icon: "list-ul"
toc_sticky: true
---

# Vanilla Neural Network Framework

[![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/) 
[![Build Status](https://travis-ci.com/nitesh4146/Vanilla-Neural-Network.svg?branch=master)](https://travis-ci.com/nitesh4146/Vanilla-Neural-Network)

* [Vanilla Source Documentation](https://htmlpreview.github.io/?https://github.com/nitesh4146/Vanilla-Neural-Network/blob/master/html/index.html)
* Install Required Packages `python3 -m pip install -r requirements.txt`
* VNN Framework: `vanilla.py`
* Usage:
    * `python3 iris_model.py`
    * `python3 cancer_model.py`
    * `python3 wine_model.py`
    * `python3 boston_model.py`

* In addition, two jupyter notebooks for classification and regression test models are included 

## Features:
1. Forward Propagation  
    ![Forward Pass](/assets/images/md_images/p-vanilla/forward.png)

    ```python
    z = input.dot(weight) + bias.T 
        if activation == "sigmoid":
            return self.sigmoid(z)
    ```
2. Loss Function  
    ![Backward Pass](/assets/images/md_images/p-vanilla/loss.png)  
    ```python
    # Cross Entropy
    y_hat_clip = np.clip(y_hat, epsilon, 1 - epsilon)
    result = ((-1.0 / (m)) * np.sum(np.sum(y_train *
                np.log(y_hat_clip), axis=1), axis=0))

    # MSE
    loss = np.square(np.subtract(y, y_pred)).mean()

    # Logistic Loss
    loss = - (y_train * np.log(y_hat) + (1 - y_train) * np.log(1-y_hat))
    result = (1.0 / m) * np.sum(loss)
    ```

3. Back Propagation  
    ![Backward Pass](/assets/images/md_images/p-vanilla/backward.png)

    ```python
    if output_layer:
        dz = out - dz_out 
    else:
        if activation == "sigmoid":
            dz = dz_out.dot(w_out.T) * self.d_sigmoid(out)
        elif activation == "tanh":
            dz = dz_out.dot(w_out.T) * self.d_tanh(out)
        elif activation == "relu":
            dz = dz_out.dot(w_out.T) * self.d_relu(out)
    ```

4. Activation Functions
    ```python
    def sigmoid(self, x):
        """
        [Private Function] Returns sigmoid function of x
        """
        return (1.0/(1.0+np.exp(-x)))

    def softmax(self, x):
        """
        [Private Function] Returns softmax function of x
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        result = exps / (np.sum(exps, axis=1, keepdims=True))

        if (np.any(np.isnan(result))):
            print("Error in Softmax")
            exit()
        return result

    def relu(self, x):
        """
        [Private Function] Returns relu function of x
        """
        return np.maximum(0.0, x)
    ```
    `Note : Currently Vanilla only supports Gradient Descent Optimizer`

5. L1 & L2 Regularization
    ```python
    def l1_reg(self, x, lam):
        """
        [Private Function] Adds L1 regularization to avoid overfitting
        """
        return (lam * np.abs(x))

    def l2_reg(self, x, lam):
        """
        [Private Function] Adds L2 regularization to avoid overfitting
        """
        return (lam * np.power(x, 2)) / 2.0
    ```
6. Misc
    * Confusion Matrix
        ```python
        cm = np.zeros(shape=(2, 2))
        for a, p in zip(y, y_pred):
            cm[int(a), int(p)] += 1
        return cm.ravel()
        ```
    * Classification Scores
        ```python
        # Sensitivity, hit rate, recall, or true positive rate
        score['sensitivity'] = tp/(tp+fn)
        # Specificity or true negative rate
        score['specificity'] = tn/(tn+fp)
        # Precision or positive predictive value
        score['precision'] = tp/(tp+fp)
        # Negative predictive value
        score['npv'] = tn/(tn+fn)
        # Fall out or false positive rate
        score['fpr'] = fp/(fp+tn)
        # False negative rate
        score['fnr'] = fn/(tp+fn)
        # False discovery rate
        score['fdr'] = fp/(tp+fp)
        # Overall accuracy
        score['accuracy'] = (tp+tn)/(tp+fp+fn+tn)
        ```

    * Model Plotter & Training Progress
    ![Training Animation](/assets/images/md_images/p-vanilla/training.gif)

## Tests
### 1. Classification: 

* #### Model Architecture  

| Dataset      | Model | 
| ----------- | ----------- |
|Iris  | `input(30) ==> L1(100) ==> L2(150) ==> L3(50) ==> out(2)`   |
|Breast Cancer | `input(4) ==> L1(100) ==> L2(60) ==> L3(60) ==> out(3)` |
|Wine | `input(13) ==> L1(200) ==> L2(120) ==> L3(60) ==> out(3)` |   


* #### Model Parameters   

| Parameter      | Iris | Breast Cancer | Wine |
| ----------- | ----------- | ----------- | ----------- |
| Learning Rate | 0.01 | 0.01 | 0.01 |
| Loss Function | Cross Entropy  | Cross Entropy | Cross Entropy |
| \# of Epochs  | 50  | 100 | 50 |
| Activation   | Sigmoid  | Sigmoid  | ReLU |


* #### Accuracy Chart  

| Set      | Iris | Breast Cancer | Wine |
| ----------- | ----------- | ----------- | ----------- |
| Training Accuracy | 100  | 90.93 | 85.25 |
| Validation Accuracy | 94.44  | 90.11 | 88.50 |
| Test Accuracy  | 95.55  | 93.86 | 87.04 |

Iris Loss Plot  
![Iris Loss Plot](/assets/images/md_images/p-vanilla/iris_loss.png)  
Cancer Loss Plot  
![Cancer Loss Plot](/assets/images/md_images/p-vanilla/cancer_loss.png)  
Wine Loss Plot  
![Wine Loss Plot](/assets/images/md_images/p-vanilla/wine_loss.png)

* #### Prediction Samples (Breast Cancer Dataset)
| Actual Labels | Predicted Labels |
| ----- | ------ |
|Benign| Malign|
|Benign| Benign|
|Benign| Benign|
|Malign| Malign|
|Malign| Malign|
|Benign| Benign|
|Benign| Benign|
|Benign| Benign|
|Malign| Malign|
|Malign| Malign|


### 2. Regression: 

* #### Model Architecture  

Boston House Prices:
`input(13) ==> L1(100) ==> L2(60) ==> L3(60) ==> out(1)`


* #### Model Parameters   

| Parameter      | Boston | 
| ----------- | ----------- | 
| Learning Rate | 0.01 |
| Loss Function | MSE  | 
| \# of Epochs  | 50  | 
| Activation   | Sigmoid + ReLU  |


* #### Accuracy Chart  

| Set      | Iris | 
| ----------- | ----------- | 
| Training MSE | 65.66  |
| Validation MSE |  49.55  | 
| Test MSE  | 45.73  |


![Iris Loss Plot](/assets/images/md_images/p-vanilla/boston_loss.png)

* #### Prediction Samples

| Actual Labels | Predicted Labels |
| ----- | ------ |
| 25.11 | 23.6 |
| 28.88 | 32.4 |
| 17.23 | 13.6 |
| 25.76 | 22.8 |
| 17.63 | 16.1 |
| 22.20 | 20. |
| 24.46 | 17.8 |
| 20.83 | 14. |
| 18.01 | 19.6 |
| 22.89 | 16.8 |
