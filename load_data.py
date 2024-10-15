import xml.etree.ElementTree as ET
import pandas as pd
from consts import *
import os


def load_xml_to_df(xml_file):
    # Parse the XML file and get the root element
    print("Processing file:", xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    drugs_and_reaction = []

    for safetyreport in root.findall('safetyreport'):
        # Iterate over the 'patient' elements
        for patient in safetyreport.findall('patient'):
            # Iterate over the 'drug' and 'reaction' elements
            for drug in patient.findall('drug'):
                for reaction in patient.findall('reaction'):
                    # Extract the 'medicinalproduct' and 'reactionmeddrapt' fields
                    medicinal_product = drug.find('medicinalproduct').text
                    reaction_meddrapt = reaction.find('reactionmeddrapt')
                    if reaction_meddrapt is not None:
                        reaction_meddrapt = reaction_meddrapt.text
                    else:
                        continue
                    drug_indication = drug.find('drugindication')
                    if drug_indication is not None:
                        drug_indication = drug_indication.text
                    else:
                        drug_indication = "unknown"
                    # Store the fields in a dictionary and append it to the list
                    drugs_and_reaction.append({DRUG: medicinal_product, REACTION: reaction_meddrapt,
                                               DRUG_TYPE: drug_indication})

    df = pd.DataFrame(drugs_and_reaction)
    print(df)
    return df


def load_data(xml_files):
    data = pd.concat([load_xml_to_df(xml_file) for xml_file in xml_files])
    return data


def collect_data(main_dir):
    xml_files = []
    for root, dirs, files in os.walk(main_dir):
       for file in files:
           if file.endswith(".xml"):
               xml_file = os.path.join(root, file)
               xml_files.append(xml_file)
    return load_data(xml_files)


def save_data(data, output_file):
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

#main_idr = r"C:\Users\itama\Downloads\2024"
# df = collect_data(main_idr)
# save_data(df, "2024.csv")
# file_n = r"C:\Users\itama\Downloads\2021\1_ADR21Q4.xml"
# df = load_xml_to_df(file_n)
# save_data(df, "2021_Q4.csv")

