import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

# Set up logging to file only (no console output)
logging.basicConfig(
    filename='error.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s'
)


def load_params(file_path: str) -> float:
    """
    Load the test size parameter from a YAML configuration file.

    Args:
        file_path (str): Path to the params.yaml file.

    Returns:
        float: Test size value for train-test splitting.
    """
    try:
        with open(file_path, 'r') as file:
            test_size = yaml.safe_load(file)['make_dataset']['test_size']
        logging.info("Loaded parameters from YAML.")
        return test_size
    except FileNotFoundError:
        logging.error(f"Parameter file not found at: {file_path}")
        raise
    except yaml.YAMLError:
        logging.error("Failed to parse YAML file. Ensure it's properly formatted.")
        raise
    except KeyError:
        logging.error("Missing key 'make_dataset -> test_size' in params.yaml")
        raise


def load_data(url: str) -> pd.DataFrame:
    """
    Load the dataset from a given CSV URL.

    Args:
        url (str): The URL pointing to the dataset.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded from URL.")
        return df
    except pd.errors.ParserError:
        logging.error("Error parsing CSV from the provided URL.")
        raise
    except Exception as e:
        logging.error(f"Failed to load data from URL: {url}\nError: {e}")
        raise


def processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dataset by removing tweet_id and filtering sentiments.

    - Keeps only 'happiness' and 'sadness' sentiments.
    - Converts 'happiness' to 1 and 'sadness' to 0.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned and processed dataset.
    """
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Data processed: filtered and labeled sentiments.")
        return final_df
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        raise
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Save the train and test datasets as CSV files.

    Args:
        data_path (str): Directory path to save CSV files.
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
    """
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logging.info(f"Data saved to {data_path}/train.csv and test.csv")
    except Exception as e:
        logging.error(f"Failed to save data to {data_path}\nError: {e}")
        raise


def main():
    """
    Main execution function to:
    - Load parameters from YAML
    - Download and process the dataset
    - Split data into train and test sets
    - Save the data to disk
    """
    try:
        test_size = load_params('params.yaml')
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = processed_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join('data', 'raw')
        save_data(data_path, train_data, test_data)
        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.error("Pipeline failed. Check above logs for details.")
        raise


if __name__ == "__main__":
    main()
