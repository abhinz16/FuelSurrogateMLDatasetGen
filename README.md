# Fuel Surrogate Machine Learning Dataset Generator

A package for generating fuel surrogate datasets diverse in UNIFAC functional group compositions for training machine learning models for predicting physicochemical properties. 
	- The dataset is created by minimizing the correlation and maximizing the entropy betweeen functional groups of surrogate mixtures.

---

## Table of Contents
1. [Process Flow](#1-process-flow)  
2. [Installation](#2-installation)  
3. [Usage](#3-usage)  
4. [License](#4-license)  
5. [Dependencies](#5-dependencies)

---
## 1) Process Flow

- Check libraries and the folders available.
- Perform fragmentation of palette components to get FG counts.
- Generate surrogate mixtures.
- Post-process the generated mixtures to ensure correct mixture IDs.
- Remove temporarily created files.
- Generate a CSV file combining the FGs of mixtures for statistical analysis later.
- Create FG distribution plots.
- Correlation and entropy analysis → how do the metrics change with the number of components and number of mixtures?
- Monte-Carlo analysis (irrespective of number of components) → how do correlation and entropy vary with number of mixtures?
- Find the minimum number of mixtures for an optimal dataset (threshold provided by the user as input at this step).
- Generate optimal dataset for the minimum number of mixtures identified using Monte-Carlo analysis.
- Finally, create the necessary files and figures.


---

## 2) Installation

Fuel Surrogate Machine Learning Dataset Generator is a Python package that runs on any platform with the proper dependencies.

### Running the Python Source (All Platforms)

- Ensure you have Python 3.10.13+ installed on your system/environment (Windows, macOS, Linux).
- Clone this repository:
```bash
git clone https://github.com/abhinz16/FuelSurrogateMLDatasetGen.git
cd FuelSurrogateMLDatasetGen
```
- Install the required dependencies (see requirements.txt).

---

## 3) Usage


- Navigate to the `FuelSurrogateMLDatasetGen` folder.

- Fill in the necessary details:
  - Locate the file: `config.py`
    - Fill in all the necessary details, especially the location to save the data (preferably the folder named `Dataset_generator`).
  - Locate the CSV files: `fuel_props.csv` and `components_with_smiles.csv`
  - Provide the UNIFAC functional group compositions of fuels in `fuel_props.csv`. Make sure to add the fuel name as well.  
    ⚠️ Do not name fuels with underscores (`_`) in them.
  - Replace the palette of components you want to use in `components_with_smiles.csv`

- Open `main.py` in a programming platform like **Spyder**.

- Run the code from a **local drive** (avoid network drives for performance and file access reasons).

- The optimized diverse dataset will be found in the `Output` folder as: `Optimal_dataset_FG_fragmentations.csv`


---


## 4) License

This project is licensed under the **MIT License**.  
Use, modify, and distribute it freely per the license terms.

---

## 5) Dependencies

- `python <= 3.10.13` and must be version 3.x  
- `pandas <= 1.5.3`  
- `matplotlib <= 3.8.3`  
- `seaborn <= 0.13.2`  
- `scipy <= 1.12.0`