"""
Sample questions for RAG system evaluation
"""

EVALUATION_QUESTIONS = [
    # --- Fact Retrieval (Specific Details) ---
    {
        "category": "Fact Retrieval",
        "question": "What is the maximum fuel flow rate allowed in the 2026 regulations?"
    },
    {
        "category": "Fact Retrieval",
        "question": "What is the minimum weight of the car for the 2026 season?"
    },
    {
        "category": "Fact Retrieval",
        "question": "Who is the current FIA President mentioned in the documents?"
    },
    {
        "category": "Fact Retrieval",
        "question": "What are the dimensions of the front wing?"
    },
    {
        "category": "Fact Retrieval",
        "question": "What is the penalty for exceeding the budget cap by less than 5%?"
    },

    # --- Summarization (High-level Overview) ---
    {
        "category": "Summarization",
        "question": "Summarize the key changes in the power unit regulations for 2026."
    },
    {
        "category": "Summarization",
        "question": "Give a brief overview of the financial regulations regarding capital expenditure."
    },
    {
        "category": "Summarization",
        "question": "What are the main safety improvements introduced in the latest technical directive?"
    },
    {
        "category": "Summarization",
        "question": "Explain the sustainability goals outlined in the strategic plan."
    },
    {
        "category": "Summarization",
        "question": "Summarize the sporting regulations concerning sprint race weekends."
    },

    # --- Complex Reasoning (Connecting multiple facts) ---
    {
        "category": "Complex Reasoning",
        "question": "How does the reduction in MGU-K energy recovery affect the overall power unit efficiency compared to the previous generation?"
    },
    {
        "category": "Complex Reasoning",
        "question": "Considering the new aerodynamic rules and tire dimensions, what is the expected impact on lap times?"
    },
    {
        "category": "Complex Reasoning",
        "question": "If a team exceeds the budget cap due to a force majeure event, what is the likely outcome based on the regulations?"
    },
    {
        "category": "Complex Reasoning",
        "question": "Compare the financial restrictions on power unit manufacturers vs. chassis constructors."
    },
    {
        "category": "Complex Reasoning",
        "question": "How do the active aerodynamics rules interact with the DRS usage protocols?"
    },

    # --- Memory & Context (Follow-up style / Specific Context) ---
    {
        "category": "Context Understanding",
        "question": "What are the 'Restricted Wind Tunnel Testing' hours?"
    },
    {
        "category": "Context Understanding",
        "question": "Define 'Curfew' in the context of race weekends."
    },
    {
        "category": "Context Understanding",
        "question": "What does the document say about 'Parc Fermé' conditions?"
    },
    {
        "category": "Context Understanding",
        "question": "Explain the 'Cost Cap' exclusions listed in Section 3."
    },
    {
        "category": "Context Understanding",
        "question": "What are the requirements for a 'Super Licence'?"
    }
]
