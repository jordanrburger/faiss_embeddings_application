import os
import json
import logging
from typing import List, Any, Dict
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import openai
from keboola.component.base import ComponentBase
from keboola.component import CommonInterface, TableDefinition
from configuration import Configuration

# Initialize logging
logging.basicConfig(level=logging.INFO)

class EmbeddingsComponent(ComponentBase):
    def __init__(self):
        super().__init__()

        # Load configuration parameters
        self.api_key = self.configuration.parameters.get('#api_key')
        self.model_name = self.configuration.parameters.get('model_name', 'distilbert-base-uncased')
        self.use_openai = self.configuration.parameters.get('use_openai', False)
        self.openai_model = self.configuration.parameters.get('openai_model', 'text-embedding-ada-002')

        # Initialize OpenAI if required
        if self.use_openai:
            openai.api_key = self.api_key

        # Initialize local transformer model and tokenizer
        if not self.use_openai:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

    def run(self):
        # Get input tables and files
        input_tables = self.get_input_tables_definitions()
        input_files = self.get_input_files_definitions()

        # Generate embeddings for tabular data
        for table in input_tables:
            df = pd.read_csv(table.full_path)
            embeddings = self.generate_embeddings(df['text_column'].tolist())  # Assuming the text column is named 'text_column'
            self.create_faiss_index(embeddings, f"{table.destination}.index")

        # Generate embeddings for files
        for file in input_files:
            with open(file.full_path, 'r') as f:
                content = f.read()
            embeddings = self.generate_embeddings([content])
            self.create_faiss_index(embeddings, f"{file.file_name}.index")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            return self.generate_openai_embeddings(texts)
        else:
            return self.generate_local_embeddings(texts)

    def generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = openai.Embedding.create(model=self.openai_model, input=texts)
        return [embedding['embedding'] for embedding in response['data']]

    def generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].tolist()

    def create_faiss_index(self, embeddings: List[List[float]], index_file_name: str):
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(index, os.path.join(self.data_path, index_file_name))

        # Save metadata for the index
        index_metadata = {
            "file_name": index_file_name,
            "dimension": dimension,
            "number_of_vectors": len(embeddings)
        }
        with open(os.path.join(self.data_path, f"{index_file_name}.json"), 'w') as f:
            json.dump(index_metadata, f)

if __name__ == '__main__':
    config = Configuration('/data/config.json')
    component = EmbeddingsComponent()
    component.run()
