import argparse
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("book_name", help="Name of the book to summarize")
args = parser.parse_args()

# Initialize the chat model
chat = ChatOpenAI(temperature=0)

# Define the system message template
system_template = "You are a knowledgeable assistant."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Define the human message template
human_template = """\
Have you heard of the book {book_name}? If so, can you summarize it?
If you have not heard of the book, reply with "I have not heard of {book_name}, summarizing from pdf...
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Build the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# Get a chat completion from the formatted messages
response = chat(
    chat_prompt.format_prompt(
        book_name=args.book_name
    ).to_messages()
)

# Print the response
print(response.content)