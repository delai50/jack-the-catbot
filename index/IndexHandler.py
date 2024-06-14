import sys
sys.path.append('../')
from pathlib import Path
from config import config
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

LOG_LEVEL = 'INDEX_GENERATOR'


class Index:
    """
    The Index class is responsible for creating and managing the search index.

    Attributes:
        index (Optional[VectorStore]): The search index.
        _data_loader (Optional[DataLoader]): The data loader.
        _data_splitter (Optional[TextSplitter]): The data splitter.
    """

    def __init__(self, docs_path, logger):
        self.logger = logger
        self.docs_path = Path(docs_path)
        self.embeddings_available = {"hf": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                                    "openai": OpenAIEmbeddings(model=config["openAIEmbeddingsModel"])
                                    }
        
        self.embedding = self.embeddings_available[config["embedding"]]
        self.vector_db = {}
        self.data_splitter = self._get_data_splitter()


    def load_data(self):
        """
        This method is responsible for generating the Documents objects
        from the data inside the self.docs_path directory.
        """
        documents = sorted(self.docs_path.glob('*'))
        data = []

        self.logger.info(f'[{LOG_LEVEL}] Loading documents...')

        for doc in documents:

            self.logger.info(f'[{LOG_LEVEL}] Loading {doc.name}...')

            # Get the file extension for choosing the right data loader.
            extension = doc.suffix

            if extension == '.pdf':
                data_loader = PyPDFLoader(file_path=doc.as_posix())
            else:
                self.logger.error(f'The file extension {extension} is not supported.')
                continue

            data.append(data_loader.load())

        return data

    def _get_data_splitter(self):
        """
        The data splitter property.

        Returns:
            TextSplitter: The data splitter.
        """
        return RecursiveCharacterTextSplitter(chunk_size=config['chunk_size'],
                                              chunk_overlap=config['chunk_overlap'], 
                                              separators=["\n\n", "\n", " ", ""]
                                             )
    def split_data(self, data):
        self.logger.info(f'[{LOG_LEVEL}] Spliting documents...')

        data_split = []
        for doc in data:
            data_split.extend(self.data_splitter.split_documents(doc))

        return data_split

    def create_index(self, documents):
        """
        Create the search index.
        """

        self.logger.info(f'[{LOG_LEVEL}] Index creation...')

        # Define the RAG db directory
        persist_dir = Path(config['ragdb_dir']) / f'rag_db_{config["embedding"]}'

        self.logger.info(f'[{LOG_LEVEL}] The persistent directory is the following one: {persist_dir}')

        # Vector database creation
        self.logger.info(f'[{LOG_LEVEL}] Creating vector database')

        data_source_path = persist_dir

        if data_source_path.is_dir():
            self.logger.info(f'[{LOG_LEVEL}] Retriever already created. Loading the existing one...')
            self.vector_db = self.load_index(data_source_path)
        else:
            self.logger.info(f'[{LOG_LEVEL}] Chromadb generation...')
            
            # Create the index
            self.vector_db = Chroma.from_documents(
                                                documents=documents,
                                                embedding=self.embedding,
                                                persist_directory=data_source_path.as_posix(), 
                                                )
            self.vector_db.persist()

    def load_index(self, persist_dir):
        vector_db = Chroma(persist_directory=persist_dir.as_posix(), 
                            embedding_function=self.embedding)
        return vector_db
