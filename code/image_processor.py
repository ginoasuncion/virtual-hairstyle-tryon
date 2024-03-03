import random
import cv2
import numpy as np
from skimage.metrics import mean_squared_error
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ImagePropertyAnalyzer:
    def __init__(self, directory: str):
        """
        Initialize the ImagePropertyAnalyzer.

        Args:
            directory (str): The directory containing the images.
        """
        self.directory = directory
        self.image_properties_list = []
        self.df = None
        
    def show_random_sample(self, num_samples: int = 5, seed: int = 42):
        """
        Display a random sample of images from the directory.
    
        Args:
            num_samples (int, optional): Number of random samples to display. Defaults to 5.
            seed (int, optional): Seed for random sampling. Defaults to 42.
        """
        # Set random seed
        random.seed(seed)
    
        # Get random sample filenames
        random_samples = random.sample(os.listdir(self.directory), num_samples)
    
        # Display each random sample image
        for filename in random_samples:
            # Load and display the image
            image_path = os.path.join(self.directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with matplotlib
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.title(filename)
            plt.axis('off')
            plt.show()

    def calculate_image_properties(self, image_path: str) -> dict:
        """
        Calculate various properties of an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            dict: A dictionary containing image properties.
        """
        # Load the image
        image = cv2.imread(image_path)

        # Shape of the image (height, width, number of channels)
        height, width, channels = image.shape

        # Total number of pixels
        total_pixels = height * width

        # Mean pixel intensity across all channels
        mean_intensity = np.mean(image)

        # Minimum and maximum pixel intensity across all channels
        min_intensity = np.min(image)
        max_intensity = np.max(image)

        # Image datatype (e.g., uint8, int16)
        data_type = image.dtype

        # Calculate sharpness for each color channel
        sharpness_values = []
        for i in range(channels):
            channel = image[:, :, i]
            laplacian_var = cv2.Laplacian(channel, cv2.CV_64F).var()
            sharpness_values.append(laplacian_var)
        avg_sharpness = np.mean(sharpness_values)

        # Calculate noise for each color channel
        noise_values = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for i in range(channels):
            channel = image[:, :, i]
            mse = mean_squared_error(channel, gray_image)
            noise_values.append(mse)
        avg_noise = np.mean(noise_values)

        # Calculate vibrancy for each color channel
        vibrancy_values = []
        for i in range(channels):
            channel = image[:, :, i]
            vibrancy_values.append(np.std(channel))
        avg_vibrancy = np.mean(vibrancy_values)

        # Return properties
        properties = {
            "image_shape": (height, width, channels),
            "total_pixels": total_pixels,
            "avg_intensity": mean_intensity,
            "avg_sharpness": avg_sharpness,
            "avg_noise": avg_noise,
            "avg_vibrancy": avg_vibrancy
        }

        return properties

    def analyze_images(self) -> pd.DataFrame:
        """
        Analyze the images in the directory and create a DataFrame containing their properties.

        Returns:
            pd.DataFrame: DataFrame containing image properties.
        """
        # Iterate through each file in the directory
        for filename in tqdm(os.listdir(self.directory)):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Consider only image files
                # Get the full path of the image
                image_path = os.path.join(self.directory, filename)

                # Calculate image properties
                properties = self.calculate_image_properties(image_path)

                # Add filename to the properties dictionary
                properties["filename"] = filename

                # Append the properties dictionary to the list
                self.image_properties_list.append(properties)

        # Create a DataFrame from the list of image properties
        df = pd.DataFrame(self.image_properties_list)
        df = df.drop_duplicates()
        # Set the filename as the index of the DataFrame
        df.set_index("filename", inplace=True)
        self.df = df
        return df

    def visualize_feature_distributions(self):
        """
        Visualize the distribution of image features using histograms.
        """
        # Calculate summary statistics for all features
        df = self.df
        summary_stats = df.describe().transpose()

        # Features to plot
        features_to_plot = ['total_pixels', 'avg_intensity', 'avg_sharpness', 'avg_noise', 'avg_vibrancy']

        # Calculate the number of subplots needed
        num_plots = len(features_to_plot)
        num_cols = 2
        num_rows = (num_plots + 1) // num_cols

        # Create subplots for each property
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))

        # Flatten the axes array if needed
        axes = axes.flatten()

        # Plot distribution for each feature
        for i, feature in enumerate(features_to_plot):
            if feature in df.columns:
                sns.histplot(data=df, x=feature, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].annotate(f"Mean: {summary_stats.loc[feature, 'mean']:.2f}\nMedian: {summary_stats.loc[feature, '50%']:.2f}\nMin: {summary_stats.loc[feature, 'min']:.2f}\nMax: {summary_stats.loc[feature, 'max']:.2f}\nStd Dev: {summary_stats.loc[feature, 'std']:.2f}",
                                 xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=10)
            else:
                fig.delaxes(axes[i])  # Delete the empty subplot

        axes.flat[-1].set_visible(False)  # to remove last plot
        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()

