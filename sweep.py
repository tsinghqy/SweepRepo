import os
import anthropic
from git import Repo
from gensim.models import Word2Vec
import tokenize
from gensim.models import Word2Vec
import io
from langchain_text_splitters import CharacterTextSplitter
from getpass import getpass
from google.colab import userdata
from langchain_anthropic import ChatAnthropic
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader


os.environ["OPENAI_API_KEY"] = userdata.get('open_ai')
os.environ["ANTHROPIC_API_KEY"] = userdata.get('anthropic_key')

llm = ChatAnthropic(model="claude-3-haiku-20240307")

django_repo_url = "https://github.com/django/django.git"
local_repo_path = "./SweepRepo/repo"

Repo.clone_from(django_repo_url, local_repo_path)

def parse_repository_files(repo_path):
    root_dir = repo_path  # Fixed variable name
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):  # Filter for Python files
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    pass
    return docs

files = parse_repository_files(local_repo_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
splits = text_splitter.split_documents(files)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

philo_template = """
You are a brilliant and meticulous engineer assigned to write code for the following Github issue. When you write code, the code works on the first try, is syntactically perfect and is fully implemented. You have the utmost care for the code that you write, so you do not make mistakes and every function and class will be fully implemented. When writing tests, you will ensure the tests are fully implemented, very extensive and cover all cases, and you will make up test data as needed. Take into account the current repository's language, frameworks, and dependencies. 
CONTEXT:

{context}

QUESTION: {question}

YOUR ANSWER:"""

philo_prompt = ChatPromptTemplate.from_template(philo_template)

retriever = vectorstore.as_retriever(k=1000)
# user_request = input("Please enter a ticket description: ")



def format_docs(docs):
    # Concatenate the page content from all loaded documents
    return docs

rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | philo_prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("""create comments for all the methods within autoreload.py and print them""")




# # from sklearn.neighbors import NearestNeighbors
# # from nltk.tokenize import word_tokenize
# # Repo.clone_from(django_repo_url, local_repo_path)

# # def parse_repository_files(repo_path):
# #     files = []
# #     for root, _, filenames in os.walk(repo_path):
# #         for filename in filenames:
# #             if filename.endswith('.py'):
# #                 file_path = os.path.join(root, filename)
# #                 with open(file_path, 'r', encoding='utf-8') as file:
# #                     content = file.read()
# #                     files.append((filename, content))
# #     return files

# # files = parse_repository_files(local_repo_path)

# # def tokenize_code(content):
# #     text = " ".join(content)

# #     # Initialize the CharacterTextSplitter
# #     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
# #     encoding="cl100k_base", chunk_size=100, chunk_overlap=0
# #     )
# #     chunks = text_splitter.split_text(text)

# #     return chunks

# # tokenized_files = [(filename, tokenize_code(content)) for filename, content in files]


# def generate_embeddings(chunks):
#     model = Word2Vec(chunks, vector_size=100, window=5, min_count=1, workers=4)
#     embeddings = [model.wv[word] for word in chunks]
#     return embeddings

# # Generate embeddings for each file
# embeddings_files = [(filename, generate_embeddings(chunks)) for filename, chunks in tokenized_files]


# embedding_store = {}

# for filename, chunks in embeddings_files:
#     if filename not in embedding_store:
#         embedding_store[filename] = []

#     embedding_store[filename].extend(chunks)

# # Take the prompt as user input
# user_request = input("Enter the prompt: ")



# def process_user_request(user_request, embedding_store):
#     # Find relevant chunks based on the user's request
#     relevant_chunks = find_relevant_chunks(user_request, embedding_store, k=5)

#     if not relevant_chunks:
#         return "No relevant code chunks found. Please try again with a different request."

#     # Pass relevant chunks to the Language Model for generating code changes
#     modified_code = generate_modified_code(user_request, relevant_chunks)

#     return modified_code

# def generate_embedding_for_request(user_request, word2vec_model, text_splitter):
#     # Split the user request into text chunks
#     text_chunks = text_splitter.split_text(user_request)

#     chunk_embeddings = []
#     for chunk in text_chunks:

#         tokens = word_tokenize(chunk)
#         tokens = [token.lower() for token in tokens]

#         chunk_embedding = calculate_average_embedding(tokens, word2vec_model)
#         chunk_embeddings.append(chunk_embedding)

#     request_embedding = calculate_average_embedding(chunk_embeddings, word2vec_model)

#     return request_embedding


# def calculate_average_embedding(embeddings, word2vec_model):
#     if embeddings:
#         avg_embedding = sum(embeddings) / len(embeddings)
#     else:
#         # If no embeddings were found, return a zero vector
#         avg_embedding = [0.0] * word2vec_model.vector_size

#     return avg_embedding

# def find_relevant_chunks(user_embedding, embedding_store, k=5):
#     # Prepare the input data for KNN
#     all_embeddings = []
#     file_names = []
#     for filename, embeddings in embedding_store.items():
#         all_embeddings.extend(embeddings)
#         file_names.extend([filename] * len(embeddings))
#     knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
#     knn_model.fit(all_embeddings)

#     distances, indices = knn_model.kneighbors([user_embedding])

#     relevant_chunks = []
#     for idx in indices[0]:
#         filename = file_names[idx]
#         chunk_idx = idx % len(embedding_store[filename])
#         relevant_chunks.append(embedding_store[filename][chunk_idx])

#     return relevant_chunks


# def generate_modified_code(user_request, relevant_chunks):
#     # placeholder
#     modified_code = "Placeholder modified code"

#     return modified_code

# # Retrieve the API key from environment variables
# api_key = ''

# # Check if the API key is provided
# if api_key is None:
#     raise ValueError("API key is not provided. Please set the ANTHROPIC_API_KEY environment variable.")

# # Initialize the Anthropogenic client with your API key
# client = anthropic.Anthropic(api_key=api_key)



# # Create a message using the model, parameters, and user message
# message = client.messages.create(
#     model="claude-3-opus-20240229",
#     max_tokens=1000,
#     temperature=0.0,
#     system="",
#     messages=[
#         {"role": "user", "content": modified_prompt}
#     ]
# )
# generated_code = message.content
# # Check if the message was successfully created
# if isinstance(generated_code, anthropic.TextBlock):
#     generated_code = generated_code.to_text()
#     # Write the generated code to a new file
# output_file_path = "output_file.py"
# with open(output_file_path, "w") as file:
#     file.write(generated_code)
# print(f"Generated code has been saved to '{output_file_path}'")