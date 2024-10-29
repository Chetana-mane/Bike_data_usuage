# Predicting London Bike Usage Using Machine Learning

## Overview

This project analyzes urban mobility patterns, forecasts bike demand, and suggests improvements for London’s bike-sharing system using data science and machine learning.

---

## Table of Contents
- [Objective](#objective)
- [Data Collection & Preparation](#data-collection--preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation & Optimization](#model-evaluation--optimization)
- [Findings and Recommendations](#findings-and-recommendations)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Objective
The aim of this project is to enhance the efficiency of London’s bike-sharing system by analyzing usage patterns, predicting demand, and providing actionable insights for operational improvements.

---

## Data Collection & Preparation

- **Bike Data**: Sourced from Transport for London (TfL) for April 2023, including trip details such as start and end stations, bike model, and trip duration.
- **Weather Data**: Integrated weather data from the Visual Crossing API, including temperature, humidity, and wind speed, to evaluate environmental impacts on bike usage.

---

## Exploratory Data Analysis

### Customer Behavior Analysis
- **Key Insights**: Observed peak usage during morning and evening rush hours, with higher rentals on weekdays and most trips lasting under 30 minutes.

   ![Figure 1: Distribution of Trip Durations](https://github.com/user-attachments/assets/3fe6449f-dd50-4640-b204-25f1c741894d)
   *Figure 1: Distribution of Trip Durations*

   ![Figure 2: Daily Trips in April 2023](https://github.com/user-attachments/assets/acdfd073-f8da-4240-82c4-f2e72d12a372)
   *Figure 2: Daily Trips in April 2023*

   ![Figure 3: Bike Rentals by Hour in April](https://github.com/user-attachments/assets/8c570dd0-841e-4d07-9572-c848bd5fbadb)
   *Figure 3: Bike Rentals by Hour in April*

   ![Figure 4: Bike Rentals by Hour - Weekdays vs Weekends](https://github.com/user-attachments/assets/999bc2e6-6244-40df-851d-ca439cd4789d)
   *Figure 4: Bike Rentals by Hour - Weekdays vs Weekends*

### Geospatial Analysis
- **Objective**: Identified high-demand stations and popular routes to optimize station placement and resource allocation.

   ![Figure 5: High-Demand Stations](https://github.com/user-attachments/assets/f40f48f3-76f1-4528-9752-af1512db10b2)
   *Figure 5: High-Demand Stations*

### Multivariate Analysis
- **Objective**: Examined the impact of factors such as day of the week, hour, and weather conditions on bike rentals.

   ![Figure 6: Hourly Rentals by Day of Week in April 2023](https://github.com/user-attachments/assets/1fe09359-ca4a-45d3-8249-dabcdf273346)
   *Figure 6: Hourly Rentals by Day of Week in April 2023*

   ![Figure 7: Trip Duration by Day of Week](https://github.com/user-attachments/assets/3b6a8268-b5cc-4e97-95cb-fd3177fb3973)
   *Figure 7: Trip Duration by Day of Week*

   ![Figure 8: Trip Duration vs. Number of Trips for Start and End Stations](https://github.com/user-attachments/assets/7359e6fe-e600-4991-9974-ec33a3237e85)
   *Figure 8: Relationship between Trip Duration and Number of Trips for Start and End Stations*

---

## Machine Learning Models

Implemented various models to predict bike usage:

- **Linear Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **Neural Networks**

A **hybrid clustering-regression technique using K-means** was applied to segment data for improved prediction accuracy.

   ![Figure 9: Model Implementation Overview](https://github.com/user-attachments/assets/e1d5d510-2171-44c6-9c01-24e93bf8a29a)
   *Figure 9: Model Implementation Overview*

   ![Figure 10: Hybrid Model with K-means Clustering](https://github.com/user-attachments/assets/999c04b0-df11-4794-8a48-47034c5669a6)
   *Figure 10: Hybrid Model with K-means Clustering*

---

## Feature Engineering

- Created new features to improve model accuracy, including interaction terms and normalization of numerical variables.

   ![Figure 11: Feature Engineering Process](https://github.com/user-attachments/assets/e8befe5a-d428-4cc1-8b8c-efd911a0d74f)
   *Figure 11: Feature Engineering Process*

   ![Figure 12: Normalized Variables](https://github.com/user-attachments/assets/a0e964d7-52a0-4eac-aa31-b01be45304ca)
   *Figure 12: Normalized Variables*

---

## Model Evaluation & Optimization

- **Evaluation Metrics**: Models were evaluated using Mean Squared Error (MSE) and R², with Neural Networks and XGBoost achieving the highest accuracy.
  
   ![Figure 13: Model Performance Comparison](https://github.com/user-attachments/assets/edf2286c-1d0d-4bfe-a4e9-4cc3e320e5e1)
   *Figure 13: Model Performance Comparison*

- **Optimization**: Hyperparameter tuning via Grid Search and Random Search improved results, especially for Random Forest and XGBoost.

---

## Findings and Recommendations

1. **Operational Improvements**: Suggested dynamic bike allocation during peak hours, enhanced resource management for high-demand stations, and integration of real-time usage data for better responsiveness.
2. **Customer Engagement**: Recommended loyalty programs, flexible pricing options, and real-time station updates to increase off-peak usage and improve user experience.
3. **Strategic Expansion**: Proposed adding stations in high-traffic areas and upgrading infrastructure on popular routes to meet demand.

---

## Limitations and Future Work

Currently, the model focuses on data from a single station (Hyde Park Corner), limiting generalization. Future work could involve expanding data collection across stations and over a longer period to develop a robust city-wide model.

---

This README summarizes the methodologies, analyses, and key insights from this project, providing a roadmap for enhancing the effectiveness of London’s bike-sharing system.
