# **Anomalous Cycle Exploration Script**
## Overview:
The Anomalous Cycle Exploration Script is a tool designed as a framework for identifying and exploring tests with anomalous cycle data. This script uses sklearn's 'LocalOutlerFactor' module to detect which cycles have anomalous cycle statistic data. It then allows you to see which cycles/cycle statistics are anomalous for a given test, plot the per cycle statistics with and without the anomalous data, and plot the timeseries data for a given cycle on the test record. The data for the script can be selected from a set of curated public datasets hosted on Voltaiq Community, or can be selected from a custom search criteria.

*Note that the current implementation does not optimize the anomaly detection using LocalOutlierFactor - this is an opportunity for improvement and can lead to false positives (e.g. detection of anomalies when none are present) for some test records.*

## Running the Script Details:
- Run the jupyter notebook code blocks from top to bottom
    - All imports are done in the first cell - all functions and analysis is contained in utils.py - this file must be located in the same directory as this Jupyter Notebook
- After running the final code block you will be prompted to choose a dataset for analysis:
    - Choose from a curated dataset, or 'Custom'
    - If 'Custom' is chosen you will need to fill in a search criteria. An example search criteria is pre-entered to help with exploration
    - Note that the first dataset is chosen by default
- Following the dataset choice, you should see 3 additional drop downs:
    - From the 'Test:' dropdown, you can select which test you wish to analyze
    - From either of the two 'Remove Cycles' dropdowns you can select to remove cycles from analysis from the begining and/or end of the test
- The table will display all cycles which contain anomalous data for one or more of the cycle statistics
    - The analytic is analyzing the following cycle statistics calculated by Voltaiq: </br>
        *Discharge Capacity*, *Charge Capacity*, *Discharge Energy*, *Charge Energy*, *Coulombic Efficiency*, *Total Charge Time*, *Total Discharge Time*, *Mean Dis. Potential by Capacity*, *Mean Ch. Potential by Capacity*
- From the 'Cycle Trace:' dropdown you can select which trace to view in the per cycle plot. 
- To view the data removed by the LOF algorithm - select the 'Anomalous \<Trace Name\>' trace from the plot's legend.
- From the 'Cycle:' dropdown you can select a cycle number to plot and explore that cycle's timeseries data

---

This repository was created by Voltaiq Community Edition. Please do not delete it or change its
sharing settings.
