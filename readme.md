# ArrakisProfiler
# Author: Tanvir Saini

ArrakisProfiler is a Python tool designed for analyzing clinical data, diversity scores, and generating scatter plots based on distance files.

## Description

ArrakisProfiler integrates various functionalities including data reading, statistical analysis, file manipulation, and visualization. It reads clinical data from a tabulated text file, calculates diversity statistics, appends these statistics to the clinical data, and generates a text file containing the combined dataset. Additionally, it identifies top and bottom entries based on diversity averages and extracts distance data for visualization. Finally, it creates scatter plots either with or without clustering, providing insights into the dataset structure.

## Workflow Overview

1. **Reading Data**: Clinical data and diversity score files are read from specified paths.
2. **Data Processing**: Diversity statistics are calculated for each unique code name in the clinical data.
3. **Data Appending**: Diversity statistics are appended to the clinical data.
4. **File Generation**: Combined data is written to a text file.
5. **Data Selection**: Top and bottom entries based on diversity averages are identified.
6. **Distance Data Extraction**: Distance data files are extracted for the selected entries.
7. **Clustering (Optional)**: If enabled, clustering is performed using the elbow method to identify optimal cluster numbers.
8. **Visualization**: Scatter plots are generated based on the extracted distance data, optionally colored by cluster.

## Functional Overview

- **read_data**: Reads clinical data and diversity score files, calculates diversity statistics.
- **append_values**: Appends diversity statistics to the clinical data.
- **generate_file**: Generates a text file containing the combined clinical data.
- **find_names**: Identifies top and bottom entries based on diversity averages.
- **find_distance**: Extracts distance data for visualization.
- **create_scatter**: Generates scatter plots based on distance data, optionally colored by cluster.
- **optimal_cluster**: Performs clustering using the elbow method to identify optimal cluster numbers.
- **arg_parser**: Sets up command-line argument parsing.
- **main**: Orchestrates the workflow based on command-line arguments.

## Dependencies

ArrakisProfiler relies on the following Python libraries:

- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib

Ensure these libraries are installed in your Python environment before running the tool.

## How to Run

To run ArrakisProfiler, run the script with the following within the parent directory:

```
python3 main.py inputfiles/clinical_data.txt inputfiles/diversity_scores/ inputfiles/distance_files/ --cluster
```

If your desired files are outside of the parent directory then run
```
python3 main.py "<path>/<to>/<your>/clinical_data.txt" "<path>/<to>/<your>/diversity_scores/" "<path>/<to>/<your>/distance_files/" --cluster
```

# Optional Arguments:
```
-l, --logging: Enable verbose logging.
-c, --cluster: Enable graphing via clusters derived from the elbow method.
```

# Error Handling


1. File Existence Verification:
    Before processing any data, ArrakisProfiler verifies the existence of specified input files and directories. 
    If any file or directory is missing, the tool promptly notifies the user and exits the workflow to prevent erroneous execution.

2. Exception Handling:
    Throughout the workflow, ArrakisProfiler employs exception handling techniques to gracefully handle unexpected errors or exceptions. 
    In the event of an error, the tool provides detailed error messages, enabling users to identify and troubleshoot issues effectively.

3. Argument Validation: 
    ArrakisProfiler validates command-line arguments provided by the user to ensure they adhere to expected formats and conventions. 
    Invalid arguments are flagged, and appropriate error messages are displayed, guiding users on correct usage.

4. Logging: 
    The tool offers optional verbose logging functionality, allowing users to enable detailed logging messages for better visibility 
    into the execution flow. This feature aids in debugging and troubleshooting, especially in complex analysis scenarios.

# Final Note: Clustering analysis
The optimal number of clusters are automatically calculated using the silhouette coefficent. If the number of
optimal clusters determined using the silhouette coefficent does not match the associated elbow plot, then additional
statistical tests may needed to determine if an optimal cluster value exists.
