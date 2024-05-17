from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_parse import LlamaParse

from dotenv import load_dotenv
load_dotenv()

def PDF_engine_create(llm):
    parser = LlamaParse(result_type="markdown", show_progress = True)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    pdf_query_engine = vector_index.as_query_engine(llm=llm)
    return pdf_query_engine