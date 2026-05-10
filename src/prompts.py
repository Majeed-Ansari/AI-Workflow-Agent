def build_prompt(context, question):

    prompt = f"""
You are an AI assistant.

Answer the user's question ONLY using the provided document context.

If the answer exists in the context, answer clearly and accurately.

If the answer is not available, say:
"I could not find the answer in the uploaded document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return prompt