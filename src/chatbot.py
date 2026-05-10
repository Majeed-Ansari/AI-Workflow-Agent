import os

from dotenv import load_dotenv
from openai import OpenAI

from src.agents import (
    research_agent,
    summary_agent,
    email_agent,
    insight_agent
)

# Load environment variables
load_dotenv()

# NVIDIA API Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


def ask_question(vector_store, question, agent_type):

    # Retrieve relevant chunks
    docs = vector_store.similarity_search(
        question,
        k=5
    )

    # Combine context
    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # Agent routing
    if agent_type == "Research Agent":

        prompt = research_agent(
            context,
            question
        )

    elif agent_type == "Summary Agent":

        prompt = summary_agent(
            context,
            question
        )

    elif agent_type == "Email Agent":

        prompt = email_agent(
            context,
            question
        )

    elif agent_type == "Insight Agent":

        prompt = insight_agent(
            context,
            question
        )

    else:

        prompt = research_agent(
            context,
            question
        )

    try:

        # NVIDIA API Response
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that answers "
                        "questions based only on provided document context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:

        return f"Error occurred: {str(e)}"