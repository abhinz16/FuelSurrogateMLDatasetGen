# Fuel Surrogate Machine Learning Dataset Generator
A package for generating fuel surrogate datasets diverse in UNIFAC functional group compositons for training machine learning models for predicting physciochemical properties.

PROCESS FLOW:
1) Check libraries and the folders available.
2) Perform fragmentation of palette components to get FG counts.
3) Generate surrogate mixtures.
4) Post processing of the generated mixtures to ensure correct mixture ids.
5) Remove temperorly created files.
6) Generating a csv file combining the FGs of mixtures for statistical analysis later.
7) Create FG distribution plots.
8) Correaltion and entropy analysis --> how does the metrics change with number oc components and number of mixtures.
9) Monte-Carlo analysis (irrespective of number of components) --> how does correlation and entropy vary with number of mixtures.
10) Find the minimum number of mixtures for an optimal dataset. Threshold provided by the user as an input at this step.
11) Generate optimal dataset for the minimum number of mixtures identified using Monte-Carlo analysis.
12) Finally, create the necessary files and figures.
    
DO NOT NAME FUELS WITH '_' IN THEM.
RUN THE CODE AND ASSOCIATED FILE FROM A LOCAL DRIVE. TRY TO AVOID NETWORK DRIVES AND SUCH.

REPLACE THE PALETTE OF COMPONENTS YOU WOULD LIKE TO USE IN 'Components_with_SMILES.csv' FILE.
PROVIDE THE UNIFAC FUNCTIONAL GROUP COMPOSITIONS OF FUEL IN 'Fuel_props.csv' FILE.
