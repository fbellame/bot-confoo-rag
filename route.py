from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer

# Route for Confoo conversation
confoo = Route(
    name="confoo",
    utterances=[        
        "Confoo est la place pour les développeurs",
        "Confoo est une importante conférence pour les développeurs à Montréal",
        "Confoo date",
        "Confoo c'est quoi?"
    ]
)

# Route for Farid
farid = Route(name="farid", utterances=[
    "Farid Bellameche est passioné d'IA",
    "Farid Bellameche va animer une session sur les chatbot avec données privées",
    "Farid"
    ]
)

# Route for technologies
technologie = Route(name="technologie", utterances=[
    "langchain",
    "streamlit",
    "openai",
    "vector database"
])

encoder = OpenAIEncoder()

routes=[confoo, farid, technologie]

dl = RouteLayer(encoder=encoder, routes=routes)