from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd

# Sample DataFrame
original_data = pd.read_csv('../resources/cmi-problematic-internet-usage/train.csv')

# Create metadata
metadata = Metadata()
metadata.detect_table_from_dataframe('table', original_data)

# Initialize the synthesizer with the metadata
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(original_data)

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=500)

# Save the synthetic data to a CSV file
synthetic_data.to_csv('../resources/synthetic_dataset.csv', index=False)
