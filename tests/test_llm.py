# from neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM
# from pydantic import BaseModel, Field
# from typing import List
# from neo4j_runway import Discovery, GraphDataModeler, PyIngest, UserInput
# import pandas as pd


# def test_llm():
#     USER_GENERATED_INPUT = UserInput(general_description='This dataset contains a list of failed banks in the United States.',
#                                  column_descriptions={
#                                             'Bank Name': 'Name of the failed bank.',
#                                             'City': 'City where the failed bank was headquartered.',
#                                             'State': 'State where the failed bank was headquartered.',
#                                             'Cert': 'FDIC certificate number of the failed bank.',
#                                             'Acquiring Institution': 'Name of the institution that acquired the failed bank.',
#                                             'Closing Date': 'Date that the failed bank closed.',
#                                             'Fund': 'FDIC fund number of the failed bank, which acts as a unique identifier in this data set.'
#                                             },
#                                 use_cases=[
#                                     "What are patterns among the locations of failed banks?",
#                                     "What are connections among the acquiring institutions of failed banks?"
#                                     ]
#                                 )
#     disc_llm = OpenAIDiscoveryLLM(base_url="http://localhost:11434/", model_name="mistral", open_ai_key="ollama")
#     failedBank_df = pd.read_csv("./tests/banklist.csv", encoding='ISO-8859-1', sep=',', on_bad_lines='skip')
#     disc = Discovery(llm=disc_llm, user_input=USER_GENERATED_INPUT, data=failedBank_df)
#     disc.run()
    
#     # modeling_llm = OpenAIDataModelingLLM()
#     # gdm = GraphDataModeler(llm=modeling_llm, discovery=disc)
#     # gdm.create_initial_model()
