
# Tiktok Project Lab: Classifying videos using machine learning**

In this project I will practice using machine learning techniques to predict on a binary outcome variable.
<br/>

**The purpose** of this project is to provide basic foundation of statistical modeling and machine learning for me and serves as a playgorund for me to try and test around different machine learning supervised algorithms. 

**The goal** of this project is to predict whether a TikTok video presents a "claim" or presents an "opinion".
<br/>

This activity follows the PACE framework, which consists of four key steps:

* **Plan** – Define objectives, set goals, and outline the approach.
* **Analyze** – Gather data, assess insights, and identify key factors.
* **Execute** – Implement the plan, take action, and monitor progress.
* **Evaluate** – Review outcomes, measure effectiveness, and refine strategies.

## Plan

* **Business need and modeling objective**

TikTok users can report videos that they believe violate the platform's terms of service. Because there are millions of TikTok videos created and viewed every day, this means that many videos get reported—too many to be individually reviewed by a human moderator.

Analysis indicates that when authors do violate the terms of service, they're much more likely to be presenting a claim than an opinion. Therefore, it is useful to be able to determine which videos make claims and which videos are opinions.

TikTok wants to build a machine learning model to help identify claims and opinions. Videos that are labeled opinions will be less likely to go on to be reviewed by a human moderator. Videos that are labeled as claims will be further sorted by a downstream process to determine whether they should get prioritized for review. For example, perhaps videos that are classified as claims would then be ranked by how many times they were reported, then the top x% would be reviewed by a human each day.

A machine learning model would greatly assist in the effort to present human moderators with videos that are most likely to be in violation of TikTok's terms of service.

* **Modeling design and target variable**

The data dictionary shows that there is a column called claim_status. This is a binary value that indicates whether a video is a claim or an opinion. This will be the target variable. In other words, for each video, the model should predict whether the video is a claim or an opinion.

This is a classification task because the model is predicting a binary class.

* **Select an evaluation metric**

There are two possibilities for bad predictions:

** False positives: When the model predicts a video is a claim when in fact it is an opinion

** False negatives: When the model predicts a video is an opinion when in fact it is a claim

In the given scenario, it's better for the model to predict false positives when it makes a mistake, and worse for it to predict false negatives. It's very important to identify videos that break the terms of service, even if that means some opinion videos are misclassified as claims. The worst case for an opinion misclassified as a claim is that the video goes to human review. The worst case for a claim that's misclassified as an opinion is that the video does not get reviewed and it violates the terms of service. A video that violates the terms of service would be considered posted from a "banned" author, as referenced in the data dictionary.

Because it's more important to minimize false negatives, the model evaluation metric will be recall.

* **Modeling workflow and model selection process**

Previous work with this data has revealed that there are ~20,000 videos in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:

1. Split the data into train/validation/test sets (60/20/20)
2. Fit models and tune hyperparameters on the training set
3. Perform final model selection on the validation set
4. Assess the champion model's performance on the test set

![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)

## Analyze

* **Structure & Size:**

The dataset contains 19,382 entries (rows) and 12 columns, each representing different attributes related to video content and performance metrics.

* **Missing Data:**

Some columns contain missing values including claim_status, video_transcription_text. However, there are very few missing values relative to the number of samples in the dataset (298 / 19382). Therefore, observations with missing values can be dropped.

* **Data Types:**

** Integers (e.g., video_id, video_duration_sec)

** Floats (engagement metrics)

** Objects (status fields like claim_status, verified_status, author_ban_status)

*  **Class Balance**

By checking class proportion, claim_status is almost evenly split between two categories:

** claim: 50.3%

** opinion: 49.7%

This near 50/50 distribution suggests that the dataset contains a balanced mix of claims and opinions.

## Construct

![](https://private-user-images.githubusercontent.com/179644177/411362403-948fbad7-b9b5-4451-8815-a94f4b7f900d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzkxNTIyNDcsIm5iZiI6MTczOTE1MTk0NywicGF0aCI6Ii8xNzk2NDQxNzcvNDExMzYyNDAzLTk0OGZiYWQ3LWI5YjUtNDQ1MS04ODE1LWE5NGY0YjdmOTAwZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIxMFQwMTQ1NDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMWMwMjY2ODdiNzcyNmU5YWExZTA1YjRhYjBiNjZhNmJiNGViNTY2ODEyNjA5ODc2NmZmY2ZiNDA4YjQyOTU1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.UM2STeEBuf1oGMgU8EdaOByicagkiqN5jv12rG_JU_w)


* **Transformation Step**
In the transformation phase, categorical variables were encoded to prepare them for model integration. Additionally, the text data contained in video_transcription_text was converted into a numerical format using the CountVectorizer technique.

* **Model Building**

A series of machine learning models were evaluated to determine the optimal predictive approach. The following algorithms were tested:

** Random Forest
** XGBoost
** CatBoost
** Support Vector Machines (SVM)
** LightGBM
** AdaBoost

Each model's performance was compared to select the best fit for the task.

## Evaluate 

![](https://private-user-images.githubusercontent.com/179644177/411378099-cfd89be1-6a68-4b1d-babb-8fa0b77357b8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzkxNTIyNDcsIm5iZiI6MTczOTE1MTk0NywicGF0aCI6Ii8xNzk2NDQxNzcvNDExMzc4MDk5LWNmZDg5YmUxLTZhNjgtNGIxZC1iYWJiLThmYTBiNzczNTdiOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIxMFQwMTQ1NDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04MTA1OGVhZmJhZjkxZjFjNzUxZTE4MTFmNDRiMWRhOGZkZThkMTA1NWQ1MzY0OGQ0YTE4ZmMyNzJlZDE4YzRmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.6EWdpAt8-xo7sDY7jvBT-wQzTNTqsNXggs7zpRtF-B8)

The accuracy scores for each tested model are as follows:

Model	Accuracy

** Random Forest (RF)	0.9995

** XGBoost (XGB)	0.9990

** CatBoost (CB)	1.0000

** Support Vector Machine (SVM)	1.0000

** LightGBM (LGBM)	0.9997

** AdaBoost (ADA)	0.9993

