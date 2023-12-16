# Import necessary libraries
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# Initialize the LLM
llm = OpenAI(temperature=0)

# Load the PDF
pdf_path = "~/Downloads/notre_dame_short.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Create a summarization chain
chain = load_summarize_chain(llm, chain_type="map_reduce")

# Run the chain on the pages to generate the summary
summary = chain.run(pages)

print(summary)
with open('summary.txt', 'w') as file:
    file.write(summary)
