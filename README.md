Fuel Surrogate Machine Learning Dataset Generator

A package for generating fuel surrogate datasets diverse in UNIFAC functional group compositions for training machine learning models for predicting physicochemical properties.
Table of Contents
1) Process Flow
2) Installation
3) Usage
4) License
5) Dependencies

1) Process Flow

a) Check libraries and the folders available.
b) Perform fragmentation of palette components to get FG counts.
c) Generate surrogate mixtures.
d) Post processing of the generated mixtures to ensure correct mixture ids.
e) Remove temporarily created files.
f) Generating a csv file combining the FGs of mixtures for statistical analysis later.
g) Create FG distribution plots.
h) Correlation and entropy analysis → how does the metrics change with the number of components and number of mixtures.
i) Monte-Carlo analysis (irrespective of number of components) → how does correlation and entropy vary with number of mixtures.
j) Find the minimum number of mixtures for an optimal dataset. Threshold provided by the user as an input at this step.
k) Generate optimal dataset for the minimum number of mixtures identified using Monte-Carlo analysis.
l) Finally, create the necessary files and figures.
2) Installation

Fuel Surrogate Machine Learning Dataset Generator is a Python package that runs on any platform with the proper dependencies. Follow these steps to set it up:
Running the Python Source (All Platforms)

    Ensure you have Python 3.10.13+ versions installed on your system/environment (Windows, macOS, Linux).

    Clone this repository:

git clone https://github.com/abhinz16/FuelSurrogateMLDatasetGen.git
cd FuelSurrogateMLDatasetGen

Install the required dependencies (see Requirements.txt in FuelSurrogateMLDatasetGen).

3) Usage

    Navigate to FuelSurrogateMLDatasetGen folder.

    First, fill in the necessary details:

        Locate csv files: fuel_props.csv and components_with_smiles.csv

        Provide the UNIFAC functional group compositions of fuel in fuel_props.csv file. Do not name fuels with _ in them.

        Replace the palette of components you would like to use in components_with_smiles.csv file.

    Next, locate main.py and open it in a programming platform like Spyder.

    Provide the filepath for FuelSurrogateMLDatasetGen folder. The generated outputs will be stored in this folder.

    Run the code and associated files from a local drive. Try to avoid network drives and such.

    The optimized diverse dataset can be found in a folder named Output with name Optimal_dataset_FG_fragmentations.csv.

4) License

This project is licensed under the MIT License.
Use, modify, and distribute it freely per the license terms.
5) Dependencies

    python <= 3.10.13 and is 3.x

    pandas <= 1.5.3

    matplotlib <= 3.8.3

    seaborn <= 0.13.2

    scipy <= 1.12.0

