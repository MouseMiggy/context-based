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
    """
    Generate semantic embedding text for livestock waste listing.
    
    Uses livestock type as reliable context anchor and incorporates user description
    to capture agricultural intent. Focuses on waste source, crop usability, and
    farming relevance. Excludes pricing, payment, delivery, and logistics.
    
    Args:
        listing_data: Dictionary containing listing information
        
    Returns:
        str: Clean, neutral English text optimized for agricultural semantic search
    """
    parts = []
    
    # 1. LIVESTOCK TYPE (Primary Context Anchor)
    # This is the most reliable field - use it as the foundation
    livestock_types = listing_data.get('livestockTypes', [])
    if livestock_types:
        # Normalize and expand livestock types for better matching
        livestock_context = []
        for livestock in livestock_types:
            normalized = normalize_livestock_type(livestock)
            livestock_context.append(normalized)
            
            # Add agricultural context for each livestock type
            if normalized in ['chickens', 'poultry']:
                livestock_context.append('poultry manure high nitrogen organic fertilizer')
            elif normalized in ['cattle', 'cow']:
                livestock_context.append('cattle manure balanced nutrients soil improvement')
            elif normalized in ['pigs', 'swine']:
                livestock_context.append('pig manure high phosphorus crop fertilizer')
            elif normalized in ['goats']:
                livestock_context.append('goat manure potassium rich vegetable fertilizer')
            elif normalized in ['rabbits']:
                livestock_context.append('rabbit manure gentle nutrients cold manure')
            elif normalized in ['buffalo', 'carabao']:
                livestock_context.append('buffalo manure organic matter paddy field fertilizer')
            elif normalized in ['horses']:
                livestock_context.append('horse manure mushroom substrate garden fertilizer')
            elif normalized in ['sheep']:
                livestock_context.append('sheep manure dry pellets organic fertilizer')
            elif normalized in ['ducks']:
                livestock_context.append('duck manure aquatic bird fertilizer')
        
        parts.extend(livestock_context)
    
    # 2. WASTE TYPE IDENTIFICATION
    # Extract waste type from name or infer from context
    name = listing_data.get('name', '').lower()
    if 'manure' in name or 'dumi' in name:
        parts.append('animal manure livestock waste organic fertilizer')
    if 'compost' in name or 'composted' in name:
        parts.append('composted organic matter soil amendment')
    if 'egg' in name or 'itlog' in name or 'shell' in name:
        parts.append('eggshells calcium rich poultry waste')
    if 'bedding' in name or 'litter' in name:
        parts.append('animal bedding livestock litter organic material')
    if 'vermi' in name or 'worm' in name:
        parts.append('vermicompost worm castings premium organic fertilizer')
    
    # 3. AGRICULTURAL INTENT FROM DESCRIPTION
    # Clean and extract farming-relevant information only
    description = listing_data.get('details', '') or listing_data.get('description', '')
    if description:
        cleaned_desc = clean_agricultural_description(description)
        if cleaned_desc:
            parts.append(cleaned_desc)
    
    # 4. GENERAL CROP SUITABILITY
    # Add general agricultural use context if not already specified
    if not any(crop in ' '.join(parts).lower() for crop in ['rice', 'corn', 'vegetable', 'fruit', 'crop']):
        parts.append('suitable for crops vegetables rice corn general farming use')
    
    # 5. ORGANIC FARMING CONTEXT
    # Emphasize organic and sustainable farming
    parts.append('organic fertilizer sustainable agriculture soil health improvement')
    
    # 6. TAGALOG-ENGLISH MAPPINGS (for multilingual search)
    # Add English equivalents for common Tagalog terms
    if 'kambing' in name or 'kanding' in name:
        parts.append('goat')
    if 'manok' in name:
        parts.append('chicken poultry')
    if 'baboy' in name:
        parts.append('pig swine')
    if 'baka' in name:
        parts.append('cattle cow')
    if 'kalabaw' in name or 'kabaw' in name:
        parts.append('buffalo carabao')
    if 'kuneho' in name:
        parts.append('rabbit')
    if 'kabayo' in name:
        parts.append('horse')
    if 'tupa' in name:
        parts.append('sheep')
    if 'pato' in name:
        parts.append('duck')
    
    # Join all parts and clean up
    semantic_text = ' '.join(parts)
    
    # Remove duplicates while preserving order
    words = semantic_text.split()
    seen = set()
    unique_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower not in seen:
            seen.add(word_lower)
            unique_words.append(word)
    
    return ' '.join(unique_words)


def normalize_livestock_type(livestock_type):
    """
    Normalize livestock type to standard format for consistent embedding generation.
    
    Args:
        livestock_type: Raw livestock type string
        
    Returns:
        str: Normalized livestock type
    """
    if not livestock_type:
        return ''
    
    normalized = livestock_type.lower().strip()
    
    # Mapping to standard types
    type_map = {
        # Cattle variations
        'cow': 'cattle',
        'cows': 'cattle',
        'cattle': 'cattle',
        'beef cattle': 'cattle',
        'dairy cattle': 'cattle',
        'dairy cow': 'cattle',
        'beef cow': 'cattle',
        'baka': 'cattle',
        
        # Buffalo/Carabao variations
        'carabao': 'buffalo',
        'water buffalo': 'buffalo',
        'buffalo': 'buffalo',
        'kalabaw': 'buffalo',
        'kabaw': 'buffalo',
        
        # Pig/Swine variations
        'pig': 'pigs',
        'pigs': 'pigs',
        'swine': 'pigs',
        'hog': 'pigs',
        'hogs': 'pigs',
        'baboy': 'pigs',
        
        # Chicken/Poultry variations
        'chicken': 'chickens',
        'chickens': 'chickens',
        'poultry': 'chickens',
        'hen': 'chickens',
        'hens': 'chickens',
        'rooster': 'chickens',
        'broiler': 'chickens',
        'layer': 'chickens',
        'manok': 'chickens',
        'quail': 'chickens',
        'pugo': 'chickens',
        'turkey': 'chickens',
        'pabo': 'chickens',
        'goose': 'chickens',
        'gansa': 'chickens',
        
        # Goat variations
        'goat': 'goats',
        'goats': 'goats',
        'kambing': 'goats',
        'kanding': 'goats',
        
        # Sheep variations
        'sheep': 'sheep',
        'tupa': 'sheep',
        
        # Rabbit variations
        'rabbit': 'rabbits',
        'rabbits': 'rabbits',
        'kuneho': 'rabbits',
        
        # Horse variations
        'horse': 'horses',
        'horses': 'horses',
        'kabayo': 'horses',
        
        # Duck variations
        'duck': 'ducks',
        'ducks': 'ducks',
        'pato': 'ducks',
        
        # Other/Others - map to general livestock
        'other': 'livestock',
        'others': 'livestock',
        'iba': 'livestock'
    }
    
    return type_map.get(normalized, normalized)


def clean_agricultural_description(description):
    """
    Clean description to extract only agricultural and farming-relevant information.
    Removes pricing, payment, delivery, and logistics details.
    
    Args:
        description: Raw description text
        
    Returns:
        str: Cleaned description focusing on agricultural intent
    """
    if not description:
        return ''
    
    # Convert to lowercase for processing
    desc_lower = description.lower()
    
    # Keywords to exclude (pricing, payment, delivery, logistics)
    exclude_keywords = [
        'price', 'cost', 'payment', 'pay', 'pesos', 'php', '₱',
        'delivery', 'shipping', 'transport', 'pickup', 'meet',
        'contact', 'call', 'text', 'message', 'whatsapp', 'viber',
        'available', 'stock', 'quantity', 'kg', 'sack', 'bag',
        'location', 'address', 'area', 'barangay', 'city',
        'negotiable', 'fixed', 'cash', 'gcash', 'bank'
    ]
    
    # Split into sentences
    sentences = description.split('.')
    
    # Keep only sentences that focus on agricultural use
    agricultural_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        
        # Skip if contains excluded keywords
        if any(keyword in sentence_lower for keyword in exclude_keywords):
            continue
        
        # Keep if contains agricultural keywords
        agricultural_keywords = [
            'crop', 'plant', 'farm', 'soil', 'fertilizer', 'organic',
            'compost', 'manure', 'nutrient', 'nitrogen', 'phosphorus',
            'potassium', 'vegetable', 'rice', 'corn', 'fruit', 'garden',
            'grow', 'harvest', 'yield', 'quality', 'rich', 'natural',
            'tanim', 'pananim', 'pataba', 'lupa', 'organiko'
        ]
        
        if any(keyword in sentence_lower for keyword in agricultural_keywords):
            agricultural_sentences.append(sentence.strip())
    
    # Join agricultural sentences
    cleaned = ' '.join(agricultural_sentences)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

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
