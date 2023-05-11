# Sepsis Prediction


Authors: Ofri Hefetz, Shai Shani Bar

## Our Task 
Our goal is to predict whether a patient in intensive care suffers from sepsis about 6 hours before being identified as suffering from sepsis,
based on clinical data about his medical condition over time. 


## What is sepsis?
Sepsis is a life-threatening medical emergency caused by your body’s overwhelming response to an infection. 
Without urgent treatment, it can lead to tissue damage, organ failure and death.

## Sepsis definition
Sepsis is your body’s extreme reaction to an infection. When you have an infection, your immune system works to try to fight it.
But sometimes your immune system stops fighting the infection and starts damaging your normal tissues and organs, leading to widespread 
inflammation throughout your body.
At the same time, an abnormal chain reaction in your clotting system can cause blood clots to form in your blood vessels. 
This reduces blood flow to the different organs of your body and can cause significant damage or even failure.

## How common is sepsis?
More than 1.7 million people in the United States receive a diagnosis of sepsis each year. There are differences in sepsis rates among different demographic groups. Sepsis is more common among older adults, with incidence increasing with each year after the age of 65 years old.

https://my.clevelandclinic.org/health/diseases/12361-sepsis


## Data
For each patient we have a file containing demographic and medical data about the patient, where each line represents data collected for one hour.
The rows are sorted according to the hours, where the first row can be referred to as the first hour of the patient's arrival at the intensive care unit and the last hour as the time when the patient left the intensive care unit for some reason


## Prediction
For the purpose of forecasting, we tested four different models. 
The models were tested based on f1 score


## Cloning Repository

git clone https://github.com/OfriHefetz/Sepsis_Prediction.git

cd Sepsis_Prediction

conda env create -f environment.yml

conda activate hw1_env

python predict.py blabla/path/test (Path to the patient tables folder)







