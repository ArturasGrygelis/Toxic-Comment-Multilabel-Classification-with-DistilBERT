# Toxic Comment Multilabel Classification with DistilBERT

**Author(s):** Artūras Grygelis

## Introduction

This project investigates the effectiveness of DistilBERT tranformers for multilabel classification of toxic comments in the imbalanced Jigsaw Toxic Comment Classification Challenge dataset. We explore the impact of different learning rates, sampling, weight initialization techniques on model performance and identify strategies for future improvement.


## Project Goals 

* Develop a multilabel classification model using DistilBERT to identify toxic comments and their specific types (toxic, severe_toxic, obscene, threat, insult, identity_hate).
* Evaluate the impact of different learning rates on model performance.
* Investigate techniques to mitigate class imbalance, including undersampling, undersampling with most unccomon values transfer to training set.
* Optimize the model for Roc_auc, F1-score and hamming loss

## Dataset

* **Jigsaw Toxic Comment Classification Challenge dataset** ([https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data))
* **Description:** Contains text comments labeled with multiple toxicity categories("toxic",	"severe_toxic",	"obscene",	"threat",	"insult",	"identity_hate"). Dataset made out of 159571 records
* **Each sentiment feature positive outcomes count :**
* ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/21140136-7fbe-462c-8649-d0b6b08b35a1)

* **Distribution of characters amount in comment texts**
* ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/f35c0765-20d4-4559-9592-cf43777a3d0a)
* **Statistical information about characters lenght in a dataset comment texts**
* ** mean        394.073221
* ** std         590.720282
* ** min           6.000000
* ** 25%          96.000000
* ** 50%         205.000000
* **75%         435.000000
* **max        5000.000000
* **  Average character lenght in English language  range from 5 to 6 characters per word when considering punctuation, spaces, and special characters. This value will vary depending on the specific data source and its characteristics. Average 65-78 words at dataset comment text  

* **Inbalance:** Only 9.6% of values belong to the minority class.
* ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/4930a468-b0c5-402b-96e9-a938fb9e9a5d)


## Methodology

**Data Preprocessing**

While sentiment analysis might consider preserving misspellings and capitalization for emotional cues, in this case focusing on identifying toxic content, we opted for a simpler text cleaning approach:

1. **Load the dataset.**
2. **Tokenization:** Convert text into word or subword units suitable for DistilBERT. This step breaks down text into meaningful units that the model can understand.
3. **Truncation :**  The truncation=True parameter  instructs the tokenizer to shorten sequences if they exceed a predefined maximum length. Max_lenght = 512 tokens. This step addresses the limitation of DistilBERT (All transformers), which can only process sequences up to a specific maximum length.
4. **Padding :** padding="max_length" ensures all sequences are padded with special tokens to reach the same length (the maximum length encountered in the batch) for efficient batch processing by the model.

**Addressing Class Imbalance:**

* **Undersampling:** Randomly undersample the majority class to balance the dataset.
    * Randomly undersampling majority class to be three times as big as minority class.
    * Randomly undersampling majority class to be three times as big as minority class and transfer most uncommon records to training set.
* **Loss function (e.g., BCEWithLogitsLoss):** Employ DistilBERT for multilabel classification with appropriate loss function (e.g., BCEWithLogitsLoss).(Weights automatically defined)
* **Manual weights in BCEWithLogitsLoss:** Define weights in BCEWithLogitsLoss manually

## Model Training

* Baseline: Train a model without modifications. ('Distilbert-base-cased' model)
* Train DistilBERT with differently sampled data.
* Train models with different learning rates (e.g., 8e-4,8e-5, 8e-6, 8e-7,) and data balancing techniques to explore their impact on convergence and performance.
* Train DistilBERT model, with automatically and manually initialized weights in BCEWithLogitsLoss.
* DistilBERT cased vs uncased. Choose best model and train DistilBERT uncased on same data, parameters.

**Evaluation**

* Use metrics like weighted ROC-AUC, weighted F1-score, and Hamming loss to assess model performance.( Weighted Roc_Auc has the biggest weight at this evaluations)
* **Main metric:** Weighted Roc_auc
* **Secondary metrics:** Weighted F1-score and hamming loss
* **Evaluation datasets:** Training data is splitted to training and validation datasets (80% , 20%). After this step, model tested on test dataset made out of 153164 records
 

## Results
* ### Baseline model test metrics ( Learning rate 0.00008):
* * Experiment name:  "driven-fire-24"
  * Train_runtime': 13473.4
  * Test loss': 0.08620832115411758,
  * Test_roc_auc': 0.8636455453693725,
  * Test_hamming_loss': 0.028694759656965416,
  * Test_f1': 0.6694065913460008,
  * Test_runtime': 585.5808,
  * Precision weighted avg : 0.60
  * Recall weighted avg : 0.77

* 
* ### Data balancing techiques:
* All sampling experiments are made with learning rate 0.00008
* ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/0f8239ba-04d4-40ec-a7de-2efb2ff40a6b)
*  **Training info**
*  ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/1a2c20ca-583d-44d3-be36-8493c64575f7)
*  **Evaluation during training info**
*  ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/0dc21554-0c2f-41d3-9f39-bdc65685145b)
* **Best Model at Sampling experiments:**
* ### Data sampling experiments, best model, test metrics ( Learning rate 0.00008):
* * Experiment name:  "classic-jazz-31"
  * Train_runtime': 7146.4 
  * Test loss': 0.09793148934841156,
  * Test_roc_auc': 0.8827655213768002,
  * Test_hamming_loss': 0.041079225150312086,
  * Test_f1':  0.616029788961539,
  * Test_runtime': 579.0439,
  * Precision weighted avg : 0.50
  * Recall weighted avg : 0.84
* * **Best Technique :** Majority class undersampling with ratio (3:1) and minority tranfer.
  * **Increased :** Higher weighted roc_auc (0.882 vs 0.863),Recall weighted avg (0.84 vs 0.77),  almost two times faster training time(7146.4 vs 13473.4).
  * **Decreased :** Weighted F1 score ( 0.616 vs 0.669), Test_hamming_loss (0.041 vs 0.028) and Precision weighted avg (0.50 vs 0.60)
* ### Learning Rate Tuning:
* All learning rate experiments are made with downsample (3:1 ratio) and minority transfer to training set from validation set
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/d5b21828-4d2f-4739-ad45-73a669edde50)
*  ## Training info
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/ed62027f-beb5-48a9-b552-4fae370e4d7d)
*  ## Evaluation during training info
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/63edcade-df71-4270-8f03-2c53ce3101d2)
* ### Learning rate experiments, best model, test metrics (Data sampling: Undersample 3:1 ratio , minority tranfer):
* * Experiment name:  "valliant-puddle-45"
  * Learning rate : 0.000008
  * Train_runtime': 7146.4398
  * Test loss':  0.10058680921792984,
  * Test_roc_auc':  0.8856993272185961,
  * Test_hamming_loss': 0.03804432773765982,
  * Test_f1': 0.6289159855057849,
  * Test_runtime': 607.4085,
  * Precision weighted avg : 0.51
  * Recall weighted avg : 0.84
* * **Best Learning rate  :** 0.000008.
  * **Increased :** Higher weighted roc_auc (0.8856 vs 0.8827),weighted F1 (0.6289 vs 0.6160), Test_hamming_loss (0.038 vs 0.410), Precision weighted avg (0.51 vs 0.50).
 
* Transformers with higher learning rates (8-5e,8-4e) tends to overfit much faster. With learning rates of (8-6e, 8-7e , training and validation loss decreases steady over time,more epochs of training can be made.)
* ### BCEWithLogitsLoss Loss function Manual vs Auto intialized weights:
* ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/a6628ca2-2fb3-4bd6-afc0-1987a8bd9cd7)
* **Training info**
*  ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/a1a218ab-2b06-4eae-82e2-41f558e4a33e)
*  **Evaluation during training info**
*  ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/87afeacf-380b-4770-a6a8-d19eeb6abe97)
* ### Auto vs Manual weights experiments, best model, test metrics (Data sampling: Undersample 3:1 ratio , minority tranfer; lr = 8-5e):
* * Experiment name:  "valliant-puddle-45"
  * Manually initialized weights didin't improve  model performance
* ### Distilbert Cased vs Uncased 
* ![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/c20f1d7b-6fb2-488d-906d-d0502dece3ea)
* **Training info**
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/4726526f-b00b-493b-b65f-1b33c6af8d5c)
*  **Evaluation during training info**
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/740527d4-db0a-4235-86b6-d44ef06b04bf)
* ### Cased vs Uncased DistilBert experiments, best model, test metrics (Data sampling: Undersample 3:1 ratio , minority tranfer):
* * Experiment name:  "worthy-forest-71"
  * Test loss':   0.09214481711387634,
  * Test_roc_auc':   0.8912378477558198,
  * Test_hamming_loss':  0.03595767295007659,
  * Test_f1':  0.6433039815795246,
  * Test_runtime': 563.5106,
  * Test_samples_per_second':   113.535,
  * Precision weighted avg : 0.53
  * Recall weighted avg : 0.85
* * **Best model :** DistilBERT uncased base transformer
  * **Increased :** Higher weighted roc_auc (0.8912 vs 0.8856), Weighted F1 (0.6433 vs 0.6289), Test_hamming_loss (0.0359 vs 0.038), Precision weighted avg (0.53 vs 0.51), Recall weighted avg (0.85 vs 0.84).
* ### Cosine vs Linear learning rate scheduler 
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/63788c06-532d-49eb-9f1e-3157cc4e8efa)
* **Training info**
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/c0ec51d7-481b-4579-b6ee-bf2140f9bb85)
*  **Evaluation during training info**
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/f4e0a259-2db3-40bf-981a-4393bc625c71)
* ### Linear vs Cosine Learning rate scheduler  experiments, best model, test metrics (Data sampling: Undersample 3:1 ratio , minority tranfer; lr = 8-5e):
* * Experiment name:  "worthy-forest-71"
  * Setting learning rate scheduller from cosine to linear, decreased model performance on a task.

# Best model
**Experiment name:**  worthy-forest-71

![Best_Model_info](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/ce9929b6-fea9-48ab-8cef-db6c0a3ad246)
### Confusion matrix of model predictions on a test set
![image](https://github.com/TuringCollegeSubmissions/argryge-DL.2.5/assets/34349260/ebedbcb6-bbbf-4ac2-b1bc-00861edbf4e9)

**Clasification report :** 
     precision    recall  f1-score   support

*           Toxic       0.45      0.96      0.61      6090
*    severe_toxic       0.42      0.42      0.42       367
*         obscene       0.57      0.85      0.68      3691
*          threat       0.52      0.45      0.48       211
*          insult       0.62      0.79      0.69      3427
*   identity_hate       0.67      0.57      0.62       712
*       micro avg       0.51      0.85      0.64     14498
*       macro avg       0.54      0.67      0.58     14498
*    weighted avg       0.53      0.85      0.64     14498
*     samples avg       0.08      0.08      0.08     14498

* Model has bad performance on severe toxic and threat classes, because of to small amout of thesse classes records. 




## Conclusion
* We successfully fine-tuned DistilBERT for multi-label toxic comment classification. Our experiments explored data augmentation, learning rates, weight initialization, and cased vs uncased models.

**Key takeaways:**

* Data augmentation with majority class undersampling and minority transfer yielded significant improvements in ROC-AUC and training speed, with a slight trade-off in F1-score and precision.
* The optimal learning rate was 8e-6, achieving the best overall performance.
* Manually initializing weights offered no benefits.
* DistilBERT uncased outperformed the cased model.
**Overall :** the DistilBERT uncased base transformer with a learning rate of 8e-6 achieved the best performance based on weighted ROC-AUC, the primary evaluation metric.


## Limitations and Future Work

**Addressing Class Imbalance:** This work explored two techniques to address class imbalance:

* Undersampling: Randomly undersampling the majority class to balance the dataset.
* Undersampling with Uncommon Records Transfer: Undersampling the majority class while transferring the most uncommon records (potentially containing valuable information) to the training set.

Future work could investigate targeted oversampling or data augmentation to further improve performance, especially for minority classes.

* **Alternative Techniques:** Exploring other pre-trained language models (e.g., RoBERTa, XLNet) or advanced data augmentation methods could be beneficial.
* **Two-Step Classification:** Implementing a two-step classification approach (toxic vs. non-toxic, then predict toxicity types) might be beneficial for handling imbalanced datasets.
* **Stop words:** Experiment by deleting stop words.
* **Hyperparameters tuning:** Tune more hyperparameters like optimizers.

**Contribution**

* Huggingface transformers.
* Kaggle notebooks
* * **Tranformers info :**
* Distilbert cased base tranformer  (https://huggingface.co/distilbert/distilbert-base-cased)
* Distilbert uncased base tranformer  (https://huggingface.co/distilbert/distilbert-base-uncased)

**Disclaimer**

* Bias and limitations: Toxic language is subjective and can be interpreted differently depending on cultural background, context, and individual perception. This model is trained on a dataset that may contain biases, and its predictions may not always reflect the nuances of human judgment. It is important to use this model responsibly and be aware of its limitations.
* Potential for misuse: Toxic comment classification models can be misused to unfairly censor or silence certain viewpoints. We encourage users to employ this model ethically and for positive purposes, such as fostering more civil online discourse.
* This disclaimer acknowledges the inherent subjectivity of toxic language and the potential for bias in the model's training data. It also highlights the importance of responsible use and avoiding misuse for censorship.
