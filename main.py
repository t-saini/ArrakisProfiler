from sklearn.cluster import KMeans
from sklearn import metrics
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import argparse
import os

def read_data(clinical_data_path:str, diversity_score_dir:str, diversity_file_ext:str = ".diversity.txt"):
    logging.info(f"Reading data from {clinical_data_path}")
    #this function takes in a string path to a clincial text file and a directory
    #containing diversity score values
    clinical_df = pd.read_csv(clinical_data_path, sep="\t")
    #the text file is opened using the read csv method 
    #also indicating that the line seperatoin is by tab
    logging.info("Successfully created DataFrame from text file")
    diversity_stats = []
    #an empty list is created and will be propigated
    #as we itterate through our last of code names
    #extracted from the clinical dataframe
    code_names = clinical_df['code_name'].unique()
    for code_name in code_names:
        #for every code name, we dynamically
        #create a string path to the file in question
        diversity_file = os.path.join(diversity_score_dir, code_name + diversity_file_ext)
        #the file is opened using pandas, and the header is set to None
        #to ensure that the first element in the text file is not turned
        #into the header for the dataframe
        logging.info(f"Reading data from {diversity_file}")
        try:
            diversity_scores = pd.read_table(diversity_file, header=None)
            #the standard deviations and mean are calculated using numpy
            diversity_mean = np.mean(diversity_scores[0])
            diversity_std = np.std(diversity_scores[0])
            #the code name, mean, and standard deviation is
            #packed into a list and appended to 
            #our initialized empty list diversity by name
            diversity_stats.append([code_name, diversity_mean, diversity_std])
        #if the file does not exist a FileNotFoundError will be
        #caught and displayed to the user
        #the workflow will exit
        except FileNotFoundError as error:
            logging.error(f"{diversity_file} not found!!!")
            logging.warning("Exiting workflow!!!")
            sys.exit(1)
        #if any other exceptio noccurs
        #the error is caught and shared with the user
        except Exception as error:
            logging.error(f"The following error occured!!!")
            logging.error(e)
            logging.warning("Exiting workflow!!!")
            sys.exit(1)
    logging.info("Successfully calculated diversity statistics")
    return clinical_df, diversity_stats

def append_values(clinical_df:pd.DataFrame, diversity_data: list)->pd.DataFrame:
    #this function takes a dataframe and a list as arguments
    #the list is converted into a dataframe
    #the dataframes are then merged, and the output is returned.
    logging.info("Appending diversity statistics to DataFrame")
    diversity_df = pd.DataFrame(diversity_data, columns=['code_name', 'averages', 'std'])
    merged_df = clinical_df.merge(diversity_df, on="code_name")
    logging.info("Successfully appended data to DataFrame")
    return merged_df

def generate_file(data:pd.DataFrame, fname:str = "clinical_data.stats.txt")->None:
    #takes in a dataframe and outputs a text file using
    #a user specified name or the default name
    logging.info(f"Attempting to write DataFrame to text file {fname}...")
    #the user is informed of the writing attempt
    try:
        data.to_csv(fname, sep="\t", index=False)
        #on success informs the users that the data is written
        logging.info(f"Successfully wrote DataFrame to {fname}")
    except Exception as e:
        #if any error occurs the user is notified and the
        #workflow exists with error code 1
        logging.error(f"An error occurred while writing to {fname}!!!")
        logging.error(e)
        sys.exit(1)

def find_names(data:pd.DataFrame, top:int = 2, bottom:int =1)->pd.DataFrame:
    #takes in a dataframe and two optional arguments top and bottom
    #and outputs a small dataframe with the data associated 
    #to the two highest and lowest averages
    sorted_data = data.sort_values(by='averages',ascending=False) 
    logging.info(f"Finding top {str(top)} value(s) based on averages")
    top_values = sorted_data.head(top)
    logging.info(f"Finding bottom {str(bottom)} value(s) based on averages")
    bottom_values = sorted_data.tail(bottom)
    selected_df = pd.concat([top_values,bottom_values])
    logging.info("Created DataFrame with upper and lower values based on averages")
    return selected_df

def find_distance(curated_data:pd.DataFrame, distance_dir:str ,file_ext:str=".distance.txt")->dict:
    #this function takes in a pandas dataframe, a string path to a directory with files
    #containing the file extension .distance.txt. the file extension to look for can
    #be changed using the optional argument file_ext
    logging.info("Extracting distance data by code name")
    code_names = curated_data['code_name'].unique()
    distance_by_name = {}
    #a list of code names are created and is itterated through
    for code_name in code_names:
        logging.info(f"Looking for distance data for {code_name}")
        #at the start of the itteration the file path to the
        #appropriate distance.txt file
        distance_file = os.path.join(distance_dir, code_name + file_ext)
        logging.info(f"Looking for {distance_file}")
        try:
            #the file is read using the read table method using
            #the comma as the seperator indicator
            logging.info(f"File found, creating DataFrame for {distance_file}")
            distance_coordinates = pd.read_table(distance_file, sep=",",header=None)
            distance_coordinates = distance_coordinates.rename(columns={0:"X", 1:"Y"})
            #the dataframe is stored as a value with the key
            #being the respective code name
            distance_by_name[code_name] = distance_coordinates
        except FileNotFoundError as error:
            logging.error(f"{distance_file} not found!!!")
            logging.warning("Exiting workflow!!!")
            sys.exit(1)
        except Exception as error:
            logging.error(f"The following error occured!!!")
            logging.error(error)
            logging.warning("Exiting workflow!!!")
            sys.exit(1)
    return distance_by_name
      

def create_scatter(pandas_dict:dict, by_cluster:bool=False)->None:
    #takes in a dictionary with pandas dataframes
    #the output are seaborn scatter plots
    code_names = pandas_dict.keys()
    logging.info(f"Creating scatter plots. Number of plots to generate: {len(code_names)}")
    #itterate along the code_names within the dictionary
    #to extrapolate the appropriate data
    for code_name in code_names:
        data = pandas_dict[code_name]
        #if clustering was called from the command line
        #the scatter plot produced will be color coded based on
        #cluster identity
        if by_cluster:
            logging.info(f"Graphing by cluster enabled. Creating scatter plot with clustering for {code_name}")
            fig = sns.scatterplot(data=data, x="X", y="Y", hue = "Cluster", palette="deep").set_title(f"{code_name} plot")
        #if the cluster argument is not used, the data is simply plotted with no
        #extra visualizatin details
        else:
            logging.info(f"Creating scatter plot for {code_name}")
            fig = sns.scatterplot(data=data, x="X", y="Y").set_title(f"{code_name} plot")
        #the plot is saved using the animal code name
        plt.savefig(f"{code_name}.png")
        #the plot is closed so a new plot object can be created
        plt.close()


def optimal_cluster(pandas_dict:dict)->dict:
    #this function takes in a dictionary with pandas dataframes
    #stored as the values, and the keys are the code names from the
    #tabulated clinical dataframe
    code_names = pandas_dict.keys()
    #cluster by name is instantiated and will hold the
    #code name as the key and the values
    #will be the updated dataframes with cluster labels
    cluster_by_name = {}
    #a list of cluster amounts is created using the range function
    cluster_range = range(1,10)
    #for each code name and dataframe
    #a KMeans calculation using the elbow method is executed
    for code_name in code_names:
        logging.info(f"Determining optimal cluster for {code_name}")
        #for every proposed cluster and its cluster label generated
        #the values are stored as key : values entries in dict
        #cluster labels
        cluster_labels = {}
        #silhouette averages contains the silhouette score for
        #every proposed cluster
        silhouette_scores = {}
        data = pandas_dict[code_name]
        x_data = list(data["X"])
        y_data = list(data["Y"])
        #the data is reshaped to be matrix like
        reshaped_data = np.array(list(zip(x_data, y_data))).reshape(len(x_data), 2)
        for cluster in cluster_range:
            #for each cluster within our list of clusters KMeans is ran, and 
            #the value of the cluster size is passed into n_clusters.
            logging.info(f"Calculating KMeans with proposed cluster {cluster} for {code_name}")
            kmean_model = KMeans(n_clusters=cluster).fit_predict(reshaped_data)
            cluster_labels[cluster] = kmean_model
            if cluster == 1:
                #if the cluster size is 1, it is skipped.
                #the silhouette_score function from sklearn requires the cluster
                #to be greater than 1
                logging.info(f"Proposed cluster size is 1, skipping silhouette score calculation")
                continue
            logging.info(f"Computing silhouette score for cluster size {cluster}")
            #the silhouette score is calculated and stored as a value with the cluster size 
            #stored as the keyf.
            silhouette_score = metrics.silhouette_score(data, kmean_model)
            silhouette_scores[cluster] = silhouette_score
        #for the given the code name, the highest silhouette value is found
        #and the associated cluster number is returned as the best cluster amount
        best_cluster = [k for k,v in silhouette_scores.items() if v == max(silhouette_scores.values())][0]
        logging.info(f"Calculated optimal cluster for {code_name}: {str(best_cluster)}")
        #using the best cluster number found, the associated cluster label is extracted and assigned
        #to the newly created column Cluster
        data["Cluster"] = cluster_labels[best_cluster]
        #the updated pandas dataframe is stored in the dictionary
        #cluster by name with the code name as they key.
        cluster_by_name[code_name] = data
    logging.info("Optimal cluster calculations by KMeans complete")
    return cluster_by_name

def arg_parser():
    #This function sets up use for command line arguments
    parser = argparse.ArgumentParser(
                        prog="ArrakisProfiler",
                        description="A tool for analyzing clinical data, diversity scores, and graphing distance files.",
                        epilog="python3 main.py inputfiles/clinical_data.txt --cluster"
                        )
    parser.add_argument('clinical_data', type=str, help='Path leading to a tabulated text file.')
    parser.add_argument('diversity_scores', type=str, help='Path leading to diversity scores directory.')
    parser.add_argument('distance_files', type=str, help='Path leading to distance files directory.')
    parser.add_argument('-c','--cluster', action='store_true', help='Enables graphing via clusters derived from the elbow method.')
    parser.add_argument('-l','--logging', action='store_true', help='Enables verbose logging messages.')
    return parser

def main(arguments:argparse.Namespace):
    #this function orchestrates the functions
    #within the file and executes the workflow
    #based on the given arguments
    input_one = arguments.clinical_data
    input_two = arguments.diversity_scores
    input_three = arguments.distance_files
    extension_check = os.path.splitext(input_one)
    if '.txt' not in extension_check:
        logging.warning("The first file path is does not lead to a text file!!!")
        logging.warning("Exiting")
        sys.exit(1)
    if not os.path.isdir(input_two):
        logging.warning("The second path provided is not a directory!!!")
        logging.warning("Exiting")
        sys.exit(1)
    if not os.path.isdir(input_three):
        logging.warning("The third path provided is not a directory!!!")
        logging.warning("Exiting")
        sys.exit(1)
    set_logging = arguments.logging
    set_cluster = arguments.cluster
    #if set_logging is True, logging will be enabled
    if set_logging:
        logging.basicConfig(level=logging.INFO, 
        format='[%(levelname)s]-%(asctime)s:::%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    #takes in the file paths and ouputs a dataframe and a list
    clinical_df, diversity_list = read_data(input_one, input_two)
    #a dataframe with the clinical data and diversity stats is created
    clinical_diversity = append_values(clinical_df, diversity_list)
    #creates a text file containing the previously created dataframe
    generate_file(clinical_diversity)
    #creates a dataframe that contains the two entries with the highest averages
    #and one entry with the lowest average.
    focused_dataframe = find_names(clinical_diversity)
    #the distance files for the associated code names from previously
    #created dataframe are extracted along with their x,y coordinates.
    data_for_plot = find_distance(focused_dataframe, input_three)
    #if set cluster is True, the data to graph will be updated
    #to include cluster identities derived from the elbow method
    if set_cluster:
        data_for_plot = optimal_cluster(data_for_plot)
    #the scatter plots are generated and based on the
    #booelan value of set_cluster may include additional
    #visual details.
    create_scatter(data_for_plot, set_cluster)

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args)
