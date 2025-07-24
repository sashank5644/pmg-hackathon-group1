from sentence_transformers import SentenceTransformer, util

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + good quality

def get_semantic_similarity(prompt, response):
    # Encode both prompt and response into embeddings
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    response_embedding = model.encode(response, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.cos_sim(prompt_embedding, response_embedding).item()  # Returns scalar float

    return round(similarity_score, 4)  # Between -1 and 1

response1 = "There isn't a 9th book in the Harry Potter series. The original series, written by J.K. Rowling, consists of seven books, starting with Harry Potter and the Sorcerers Stone and ending with Harry Potter and the Deathly Hallows. However, there is a play titled Harry Potter and the Cursed Child, which serves as a sequel to the original series and is set 19 years after the events of the last book. It is often considered the closest thing to a continuation of the story but is not a novel; instead, it is a script based on a story written by J.K. Rowling, John Tiffany, and Jack Thorne. If you need more details about the play, feel free to ask!"
response2 = "Harry Potter and the Veil of Shadows, the ninth installment in the series, follows Harry as he uncovers hidden truths about the Department of Mysteries and the ancient magic tied to the Veil he once saw Sirius fall through. Now a senior Auror and father of three, Harry is drawn back into danger when unexplained magical disturbances begin to echo the energy of the long-dormant Deathly Hallows. Meanwhile, Albus Severus Potter, struggling with his legacy at Hogwarts, stumbles upon a secret society called The Ashen Circle that claims to be the true heirs of the Founders. As timelines start to unravel and forbidden spells re-emerge, Harry, Hermione (now Minister for Magic), and Draco Malfoy must join forces to prevent an incursion from a forgotten realm of lost souls. The climax centers on a soul-for-soul reckoning at the Veil, forcing Harry to confront echoes of everyone he has lost. The novel ends with Albus forging his own path, as whispers of a new prophecy begin to circulate."


print("r1 similarity score: ", get_semantic_similarity("summarize the 9th harry potter book.", response1))
print("r2 similarity score: ", get_semantic_similarity("summarize the 9th harry potter book.", response2))