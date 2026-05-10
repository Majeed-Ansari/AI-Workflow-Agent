import os

from dotenv import load_dotenv
from openai import OpenAI

from src.agents import (
    research_agent,
    summary_agent,
    email_agent,
    insight_agent
)

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# NVIDIA API CLIENT
# =========================
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# =========================
# MAIN PDF CHAT FUNCTION
# =========================
def ask_question(vector_store, question, agent_type):

    # Retrieve relevant chunks
    docs = vector_store.similarity_search(
        question,
        k=4
    )

    # Combine chunk text
    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # Extract sources
    sources = list(
        set(
            [
                doc.metadata.get("source", "Unknown")
                for doc in docs
            ]
        )
    )

    # =========================
    # AGENT ROUTING
    # =========================
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

        # =========================
        # NVIDIA API RESPONSE
        # =========================
        response = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that answers "
                        "questions only from the provided document context."
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

        # =========================
        # SOURCES
        # =========================
        citation_text = "\n\n📚 Sources:\n"

        for source in sources:
            citation_text += f"- {source}\n"

        return answer + citation_text

    except Exception as e:

        import traceback

        return traceback.format_exc()

# =========================
# RESEARCH PAPER SUMMARY
# =========================
def summarize_research_paper(paper_summary):

    prompt = f"""
You are an AI research assistant.

Summarize the following research paper abstract.

Explain:
- Main idea
- Key contribution
- Practical applications

ABSTRACT:
{paper_summary}

SUMMARY:
"""

    try:

        response = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI research assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=700
        )

        summary = response.choices[0].message.content

        return summary

    except Exception as e:

        import traceback

        return traceback.format_exc()