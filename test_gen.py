from anki_gen import generate_cards, build_client
from config import load_config

cfg = load_config()
client = build_client(cfg["gemini_api_key"])
 
text = "The choroid plexus produces CSF. It is located in the ventricles."
 
cards = generate_cards(
    chunk       = text,
    deck        = "Test",
    client      = client,
    chunk_index = 1,
    chunk_total = 1,
    topic       = "CSF production",
)
 
print(f"Cards: {len(cards)}")
for c in cards:
    print(c)