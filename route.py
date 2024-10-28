from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer

# Route for Confoo conversation
conference = Route(
    name="/dev/mtl",
    utterances=[        
        "dev mtl",
        "/dev/mtl",
        "/dev/mtl2024",
        "conf",
        "conference"
    ]
)

# Route for Farid
farid = Route(name="farid", utterances=[
    "Farid Bellameche",
    "Farid"
    ]
)

# Route for Joke
joke = Route(name="joke", utterances=[
    "Joke",
    "Blague",
    "Amusement"
    ]
)

encoder = OpenAIEncoder()

routes=[conference, farid, joke]

dl = RouteLayer(encoder=encoder, routes=routes)