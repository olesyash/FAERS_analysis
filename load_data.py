import xml.etree.ElementTree as ET
import pandas as pd
from consts import *


def load_xml_to_df(xml_file):
    # Parse the XML file and get the root element
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
                    reaction_meddrapt = reaction.find('reactionmeddrapt').text
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







# data_path = "2_ADR24Q2.xml"
# data2 = pandas.read_xml(data_path)
# print(data2.columns)
# data_path = "3_ADR24Q2.xml"
# data3 = pandas.read_xml(data_path)
# print(data3.columns)