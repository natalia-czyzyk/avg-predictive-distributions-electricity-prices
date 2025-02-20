# Averaging predictive distributions across calibration windows for day-ahead electricity prices

This repository contains the code, data, and analysis scripts for replicating the results from the article "Averaging Predictive Distributions Across Calibration Windows for Day-Ahead Electricity Price Forecasting" by Tomasz Serafin, Bartosz Uniejewski, and Rafał Weron.

## Project Overview

This project aims to reproduce the findings of the original study, which explores methods for improving probabilistic electricity price forecasting by averaging predictive distributions across different calibration windows. The study examines two techniques: **Quantile Regression Averaging (QRA)** and **Quantile Regression Machine (QRM)**, both of which enhance forecast accuracy by leveraging multiple calibration windows.

The research utilizes datasets from two major power markets:
- **Nord Pool**: A hydropower-dominated market with strong seasonal variations.
- **PJM Interconnection**: The largest competitive wholesale electricity market in the U.S., with a balanced coal-gas-nuclear energy mix.

## Key Components

- **Data Analytics**: Implemented in Python for preprocessing, modeling, and visualization.
- **Market Data**: Includes datasets from Nord Pool and PJM.
- **Replication of Results**: Step-by-step execution of scripts to reproduce findings from the original article.

## Project Structure

All files are organized into the following directories:

- **data/** – Contains datasets from Nord Pool and PJM, along with the original article.
- **data_analysis/** – Python scripts for data processing and analysis.
- **results/** – Includes reports and presentations summarizing findings.

## Setup and Usage

### Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `statsmodels`

### Running the Analysis

1. **Run the Model Files:**
   ```bash
   python data_analytics/*_model.py
   ```
2. **Run the Quantile Regression Files:**
   ```bash
   python data_analytics/*_quantile_*.py
   ```
3. **Generate Charts:**
   ```bash
   python data_analytics/charts.py
   ```
4. **Final Analysis (APS21):**
   ```bash
   python data_analytics/NP2018-QRM-APS21.py
   ```

## Results

The analysis helps in understanding:
- The accuracy of the reproduced results in comparison to the original study.
- The impact of different market factors on electricity prices.
- Statistical relationships captured using quantile regressions.

Final results and insights are documented in the **results/** folder.

## Contributors
- **Natalia Czyżyk**
- **Weronika Urbańczyk** 

## Acknowledgments
- Authors of "Averaging Predictive Distributions Across Calibration Windows for Day-Ahead Electricity Price Forecasting" for the original study.
- Data providers: Nord Pool and PJM.
- Open-source tools: Python, Pandas, Matplotlib, Numpy.

