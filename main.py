import ollama
import json
import os
import numpy as np

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "qwen2.5:3b"

#numpy binary
VECTORS_FILE = 'vector.npy'

#chunks file
CHUNKS_FILE  = 'chunks.json'


CHUNKS = []
MATRIX = None
DISCLAIMER = "This is general information, not medical advice. Please consult a doctor or pharmacist."
TOP_N = 3
BATCH_SIZE = 128


REFUSAL = "I don't have information about that in my knowledge base."

SYSTEM_PROMPT = f"""You are a medicine information assistant. You answer from label text that is given to you.

Rules:
1. Use only the label text in the CONTEXT. Never use outside knowledge.
2. If the CONTEXT does not clearly answer the question, or it is about a different medicine than the one asked, reply with exactly this sentence and nothing else: {REFUSAL}
3. Start your answer with the medicine name.
4. Report what the label says, including doses and limits. Do not tailor it to the person and do not add advice.
5. Answer in 1 to 3 short sentences. If the label text ends abruptly, give only what it says.
"""



def load():

    chunks = []
    with open('medicines.jsonl', encoding='utf-8') as file:
        for line in file:
            clean_line = line.strip()

            if clean_line:

                record = json.loads(clean_line)

                value = record.get("chunk")


                if value is not None:
                    chunks.append(value)
    return chunks


#embed in batches (128 iin one go)
def embed_in_batches(chunks):
    embeddings = []

    for st in range(0, len(chunks), BATCH_SIZE):
        data = chunks[st: st+BATCH_SIZE]
        response = ollama.embed(model = EMBEDDING_MODEL, input = data)
        embedding = response['embeddings']
        embeddings.extend(embedding)

    return embeddings

        




#indexing

def indexing():
    global CHUNKS, MATRIX
    try:
        #checking for saved files
        if os.path.exists(VECTORS_FILE) and os.path.exists(CHUNKS_FILE):
            MATRIX = np.load(VECTORS_FILE)
            with open(CHUNKS_FILE, encoding='utf-8') as f:
                CHUNKS = json.load(f)
            print(f"Loaded {len(CHUNKS)} chunks from disk")
            return
        

        # elsee building and saving it
        chunks = load()
        embeddings = embed_in_batches(chunks)
        matrix = normalize_rows(np.array(embeddings, dtype=np.float32))
        np.save(VECTORS_FILE, matrix)

        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False)


        CHUNKS, MATRIX = chunks, matrix

        print(f"Built and saved {len(CHUNKS)} chunks")

    except Exception as e:
        print(f"Indexing failed: {e}")


#normalization function 
def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


'''
#REPLACED BY NUMPY SIMILARITY

#to find similarity
def cosine_similarity(a, b):


    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)


 '''

#returning top n chunks that matches with user's query
def retrieve(q):
    try:
        q_embed = ollama.embed(model=EMBEDDING_MODEL, input = q)['embeddings'][0]
    except Exception as e:
        print(f"AN ERROR OCCURED!, QUESTION EMBEDDING FALIED!: {e}")
        return []


    q_vec = np.array(q_embed)

    #making length 1
    q_vec = q_vec / np.linalg.norm(q_vec)

    #matrix multiply
    scores = MATRIX @ q_vec

    #the top-n results
    best_rows = np.argsort(scores)[::-1][:TOP_N]

    results = []
    for i in best_rows:
        results.append((CHUNKS[i], float(scores[i])))
    return results


#designing the answer structure
def build_user_message(retrieved, q):
    
    context = "\n".join(chunk for chunk, score in retrieved)
    return f"CONTEXT:\n{context}\n\nQUESTION: {q}\n If the CONTEXT does not answer this, reply exactly: {REFUSAL}"

def answer(q):
    if q == '':
        print("Enter a valid input!")
        return
    retrieved = retrieve(q)

    if not retrieved:
        print("retrieve failed!")
        return

    '''
    MIN_SCORE = 0.5
    if retrieved[0][1] < MIN_SCORE:
      print("I don't have information about that in my knowledge base.")
      return

    print("The chunks are retrieved~")
    '''
    #for chunk, score in retrieved:
     #   print(f"score: {score:.2f} - {chunk}")

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_message(retrieved, q)},
        ],
        stream=True,
        options={'temperature': 0, 'num_ctx': 4096},
    )

    print("Answer: ")
    for p in stream:

        #to print in a single line
        print(p['message']['content'], end = '', flush = True)
    print()
    print(DISCLAIMER)

    print("Sources: ")
    

    for chunk, score in retrieved:
        label = chunk.split(': ', 1)[0]
        print(f"- {label} - score : {score:.2f}")







if __name__ == '__main__':
    indexing()

    if MATRIX is None:
        print("Index failed!.")
        exit()

    print("Index is ready~")

    while True:
        query = input('\nEnter your questions: ').strip()
        if query.lower() == 'quit':
            break
        answer(query)
