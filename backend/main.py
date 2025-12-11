from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import firestore
from google.oauth2 import service_account
from dotenv import load_dotenv
import asyncio

load_dotenv()

app = FastAPI(title="AgriLink Semantic Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize MPNet model (multilingual version)
print("Loading multilingual MPNet model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
print("Multilingual model loaded successfully!")

# Initialize Firestore
service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
service_account_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
project_id = os.getenv('FIRESTORE_PROJECT_ID')

if service_account_json:
    # For Render deployment - credentials from environment variable JSON string
    credentials_dict = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    db = firestore.Client(credentials=credentials, project=project_id)
elif service_account_path and os.path.exists(service_account_path):
    # For local development - credentials from file
    credentials = service_account.Credentials.from_service_account_file(service_account_path)
    db = firestore.Client(credentials=credentials, project=project_id)
else:
    # Fallback to default credentials
    db = firestore.Client(project=project_id)

def create_semantic_text(listing_data):
    """Create semantic search text from listing data"""
    parts = []
    
    # Add name
    if 'name' in listing_data:
        parts.append(listing_data['name'])
    
    # Add description or details (check both for compatibility)
    if 'details' in listing_data:
        parts.append(listing_data['details'])
    elif 'description' in listing_data:
        parts.append(listing_data['description'])
    
    # Add category
    if 'category' in listing_data:
        parts.append(listing_data['category'])
    
    # Add location
    if 'ownerLocation' in listing_data:
        parts.append(listing_data['ownerLocation'])
    elif 'location' in listing_data:
        parts.append(listing_data['location'])
    
    # Add breed if available
    if 'breed' in listing_data:
        parts.append(listing_data['breed'])
    
    # Add age if available
    if 'age' in listing_data:
        parts.append(str(listing_data['age']))
    
    return ' '.join(parts)

class EmbedRequest(BaseModel):
    text: str

class ListingEmbedRequest(BaseModel):
    listingId: str

class SearchRequest(BaseModel):
    text: str
    top_k: int = 10

class EmbedResponse(BaseModel):
    embedding: List[float]

class SearchResult(BaseModel):
    id: str
    score: float
    data: dict

class SearchResponse(BaseModel):
    matches: List[SearchResult]

@app.get("/")
async def root():
    return {"message": "AgriLink Semantic Search API is running"}

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(request: EmbedRequest):
    """Convert text to MPNet embedding"""
    try:
        embedding = model.encode(request.text, convert_to_numpy=True)
        return EmbedResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed-listing")
async def embed_listing(request: ListingEmbedRequest):
    """Generate and store embedding for a specific listing"""
    try:
        # Get listing from Firestore
        doc_ref = db.collection('livestock_listings').document(request.listingId)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        listing_data = doc.to_dict()
        
        # Create semantic text
        semantic_text = create_semantic_text(listing_data)
        
        if not semantic_text.strip():
            raise HTTPException(status_code=400, detail="No content to embed")
        
        # Generate embedding
        embedding = model.encode(semantic_text, convert_to_numpy=True)
        
        # Update document with embedding
        doc_ref.update({
            'mpnet_embedding': embedding.tolist()
        })
        
        return {"message": "Embedding generated and stored successfully", "listingId": request.listingId}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Search livestock listings by semantic similarity"""
    try:
        # Get query embedding
        query_embedding = model.encode(request.text, convert_to_numpy=True)
        
        # Get all listings from Firestore
        listings_ref = db.collection('livestock_listings')
        docs = listings_ref.stream()
        
        matches = []
        
        for doc in docs:
            listing_data = doc.to_dict()
            
            # Check if listing has embedding
            if 'mpnet_embedding' not in listing_data:
                continue
                
            stored_embedding = np.array(listing_data['mpnet_embedding'])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            matches.append({
                'id': doc.id,
                'score': float(similarity),
                'data': listing_data
            })
        
        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x['score'], reverse=True)
        top_matches = matches[:request.top_k]
        
        return SearchResponse(matches=top_matches)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
