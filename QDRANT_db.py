import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import CollectionStatus, UpdateStatus

load_dotenv('./.env')



class QdrantDB:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
    
    def create_collection(self, collection_name, vector_size):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        collection_name = self.client.get_collection(collection_name)
        self.client.close()
        return collection_name
    
    def add_data_into_collection(self, collection_name, points):
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        output = {'count': self.client.count(collection_name=collection_name), 'status': operation_info.status}
        self.client.close()
        return output
    
    def delete_collection(self, collection_name):
        try:
            self.client.delete_collection(collection_name=collection_name)
            self.client.close()
            return f"Collection {collection_name} deleted successfully"
        except Exception as e:
            return f"Error deleting collection {collection_name}: {str(e)}"
        