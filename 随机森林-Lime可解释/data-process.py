import pandas as pd
import numpy as np
import os


def load_and_process_excel(file_path):
    """
    Load an Excel file, preprocess it, and convert it to a one-dimensional numpy array.
    Include or exclude the first column as part of the feature vector.
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl', header=None)  # No header assumed
        if df.empty:
            print(f"Warning: Empty dataframe loaded from {file_path}. Skipping.")
            return None

        df.fillna(df.mean(), inplace=True)  # Handle missing values by filling them with the mean of each column

        # Include the first column as part of the feature vector
        flattened_data = df.to_numpy().flatten()  # Flatten the entire DataFrame

        return flattened_data
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


def process_directory(directory_path):
    """
    Process all subdirectories in a given directory.
    Each subdirectory contains multiple Excel files.
    Return a list of lists where each inner list represents a row in the CSV.
    """
    all_data = []
    for subdir_name in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir_name)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.xlsx'):
                    file_path = os.path.join(subdir_path, filename)
                    vector = load_and_process_excel(file_path)
                    if vector is not None:
                        # Insert folder name as the first element
                        vector = np.insert(vector, 0, subdir_name)
                        all_data.append(vector)
                        print(f"Processed {filename}")
                    else:
                        print(f"Skipped {filename} due to an error.")
    return all_data


def save_list_to_csv(data_list, output_file):
    """
    Save a list of lists to a CSV file.
    """
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False, header=False)
    print(f"Data saved to {output_file}")


# Main execution part of the script
if __name__ == "__main__":
    directory_path = 'D:\\所有文档\\项目\\CD-数据集hzj\\12色\\12色\\Ag分类1与0分类'  # Path to the directory containing subdirectories with Excel files
    output_file = 'data_save/Ag.csv'  # Output file name for the resulting CSV

    # Process all subdirectories in the specified directory
    data_list = process_directory(directory_path)

    # Print the final data matrix
    if data_list:
        #print("Processed data:")
        #print(data_list)

        # Save the data list to a file
        save_list_to_csv(data_list, output_file)
    else:
        print("No valid Excel files processed.")
