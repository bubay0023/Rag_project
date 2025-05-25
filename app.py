import pandas as pd
from gemini_text_embedding import get_text_embedding
from QDRANT_db import QdrantDB


def main(data):
    qdrant_db = QdrantDB()
    collection_name = "itsm_collection"
    vector_size = 3072
    collection = qdrant_db.create_collection(collection_name, vector_size)
    points =[]
    for i,row in data.iterrows():
        embedding = get_text_embedding(row['description'])
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
    output = qdrant_db.add_data_into_collection(collection_name, points)
    print(output)







if __name__ == "__main__":
    data = pd.read_csv('/home/bubay/Desktop/Gen AI/dataset/ITSM_data.csv')
    main(data)
