import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str

def excel_engine_create(llm,excel_path):
    population_path = os.path.join("data", excel_path)
    population_df = pd.read_excel(population_path)

    population_query_engine = PandasQueryEngine(
        df=population_df, verbose=True, instruction_str=instruction_str,llm = llm
    )
    population_query_engine.update_prompts({"pandas_prompt": new_prompt})
    return population_query_engine