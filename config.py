config = {}

# Embedding and LLM model to use
config['openAIEmbeddingsModel'] =  "text-embedding-3-large"
config['openAIModel'] = "gpt-3.5-turbo-0125"

# Model hp
config['temperature'] = 0

# Chunk hps
config['chunk_size'] = 1500
config['chunk_overlap'] = 150

# Number of retrieved docs
config['k_retrieved_chunks'] = 2
config['token_memory_limit'] = 1500

# Path to the documents and persistent directory
config['rag_docs'] = "../docs"
config['ragdb_dir'] = "../persistent_dir"

# Embedding type
config['embedding'] = "openai"
