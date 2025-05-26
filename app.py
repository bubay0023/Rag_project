import pandas as pd
from text_embedding import EmbeddingModel
from QDRANT_db import QdrantDB
from time import sleep


def main(data):
    qdrant_db = QdrantDB()
    collection_name = "itsm_collection_with_gemini"
    #vector_size = 3072
    #collection = qdrant_db.create_collection(collection_name, vector_size)
    points =[]
    for i,row in data.head(10).iterrows():
        emb = EmbeddingModel()
        embedding = emb.get_gemini_text_embedding(row['description'])
        points.append({
            "id": i + 1,
            "vector": embedding,
            "payload": {
                "id": row['id'],
                "description": row['description'],
                "answer": row['answer'],
                "type": row['type'],
                "priority": row['priority'],
                "queue": row['queue']
                }
        })
        sleep(10)
    output = qdrant_db.add_data_into_collection(collection_name, points)
    print(output)




def main_ollama(data):
    qdrant_db = QdrantDB()
    collection_name = "itsm_collection"
    #vector_size = 768
    #collection = qdrant_db.create_collection(collection_name, vector_size)
    points =[]
    j=1 
    for i,row in data.iterrows():
        j += 1
        emb = EmbeddingModel()
        embedding = emb.get_ollama_text_embedding(row['description'])
        points.append({
            "id": i + 1,
            "vector": embedding,
            "payload": {
                "id": row['id'],
                "description": row['description'],
                "answer": row['answer'],
                "type": row['type'],
                "priority": row['priority'],
                "queue": row['queue']
                }
        })
        if j == 10:
            j = 1 # Reset j after every 100 entries
            output = qdrant_db.add_data_into_collection(collection_name, points)
            print(output)
            points.clear() # Clear the points list after every 100 entries
            


def search_similar(data, query, limit):

    qdrant_db = QdrantDB()
    collection_name = "itsm_collection"
    emb = EmbeddingModel()
    embedding = emb.get_ollama_text_embedding(query)
    search_result = qdrant_db.Qdrant_search(collection_name, embedding, limit)
    
    results = []
    for hit in search_result:
        results.append(hit.payload)
    
    return search_result#results

if __name__ == "__main__":
    #main(pd.read_csv('/home/bubay/Desktop/Gen AI/dataset/ITSM_data.csv'))
    #main_ollama(pd.read_csv('/home/bubay/Desktop/Gen AI/dataset/ITSM_data.csv'))
    while True:
        data = pd.read_csv('/home/bubay/Desktop/Gen AI/dataset/ITSM_data.csv')
        INC = input("Enter your INC (or type 'exit' to quit): ")
        if INC.lower() in ['exit', 'quit',"q"]:
            break
        data = data[data['id'] == int(INC)]
        if data.empty:
            print(f"No data found for INC {INC}. Please try again with new inc.")
            continue
        results = search_similar(data, INC, limit=2)
        print("Search Results:")
        for result in results:
            print(result)
        print("\n")

    


