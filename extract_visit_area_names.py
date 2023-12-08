import pandas as pd

# Set the region constant
REGION = '도서지역'

# Read the dataset from the specified region
dataset = pd.read_csv(f"dataset/data_after_preprocessing/{REGION}/dataset.csv")

# Extract unique visit area names from the 'VISIT_AREA_NM' column and store them in a list
visit_area_names = dataset["VISIT_AREA_NM"].unique().tolist()

# Create a dictionary with VISIT_ID as the key and VISIT_AREA_NM as the value
visit_area_dict = {}
for i in range(len(dataset)):
    visit_area_dict[dataset.iloc[i]["VISIT_ID"]] = dataset.iloc[i]["VISIT_AREA_NM"]

# Save the above dictionary to a CSV file (encoded in utf-8), where the first column is 'key' and the second column is 'value'
import csv
with open(f"visit_area_dict_{REGION}.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "value"])
    for key, value in visit_area_dict.items():
        writer.writerow([key, value])