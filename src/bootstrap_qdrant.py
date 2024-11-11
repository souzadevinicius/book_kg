from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Retrieve embeddings and metadata
results = client.search(
    collection_name="your_collection",
    query_vector=[0] * embedding_dim,  # Dummy query to retrieve all
    limit=10000  # Adjust based on your collection size
)

# Write embeddings to TSV
with open("embeddings.tsv", "w") as f:
    for point in results:
        f.write("\t".join(map(str, point.vector)) + "\n")

# Write metadata to TSV
with open("metadata.tsv", "w") as f:
    for point in results:
        f.write(str(point.payload.get("label", "")) + "\n")

# Create configuration file
config = {
    "embeddings": [
        {
            "tensorName": "My Qdrant Embeddings",
            "tensorShape": [len(results), embedding_dim],
            "tensorPath": "embeddings.tsv",
            "metadataPath": "metadata.tsv"
        }
    ]
}

import json
with open("config.json", "w") as f:
    json.dump(config, f)