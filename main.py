from dotenv import load_dotenv

from config import *

from prompts import context
from note_engine import note_engine
from excel_parse_engine import excel_engine_create
from pdf_engine import PDF_engine_create
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama

load_dotenv()

llm = Ollama(model="mistral",request_timeout = 30.0)

excel_query_engine = excel_engine_create(llm,EXCEL_FILE_NAME)
pdf_query_engine = PDF_engine_create(llm)

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=excel_query_engine,
        metadata=ToolMetadata(
            name="chemical_data",
            description="this gives information about the properties of organic chemical compounds",
        ),
    ),
    QueryEngineTool(
        query_engine=pdf_query_engine,
        metadata=ToolMetadata(
            name="organic_chem_data",
            description="this contains information about organic chemistry, nomenclature, chemical propeties and classification of organic compounds"
        ),
    ),
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
