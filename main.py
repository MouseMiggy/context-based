from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
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

@app.post("/crop-waste-compatibility")
async def analyze_crop_waste_compatibility(request: CropWasteCompatibilityRequest):
    """Analyze crop-waste compatibility using MPNet and knowledge base"""
    try:
        # Get base knowledge for the waste type
        waste_type_key = None
        for key in crop_waste_knowledge:
            if key.lower() in request.wasteType.lower() or request.wasteType.lower() in key.lower():
                waste_type_key = key
                break
        
        if not waste_type_key:
            # If exact match not found, try to find the best match using MPNet
            waste_types = list(crop_waste_knowledge.keys())
            waste_embeddings = model.encode(waste_types, convert_to_numpy=True)
            query_embedding = model.encode(request.wasteType, convert_to_numpy=True)
            
            similarities = []
            for i, waste_type in enumerate(waste_types):
                similarity = np.dot(query_embedding, waste_embeddings[i]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(waste_embeddings[i])
                )
                similarities.append((waste_type, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            waste_type_key = similarities[0][0] if similarities[0][1] > 0.5 else "Cattle Manure"
        
        base_knowledge = crop_waste_knowledge[waste_type_key]
        
        # If specific crop category is requested, rank crops within that category
        if request.cropCategory:
            category_crops = {
                "vegetables": ["Leafy Vegetables", "Vegetables (all types)", "Vegetable Gardens", "Tomatoes", "Peppers", "Eggplant", "Broccoli", "Cabbage", "Cauliflower"],
                "fruits": ["Fruit Trees", "Orchard Fruits", "Berry Crops", "Grapes", "Strawberries"],
                "grains": ["Rice", "Corn", "Wheat"],
                "root_crops": ["Root Crops"],
                "legumes": ["Beans", "Peas"]
            }
            
            category = request.cropCategory.lower()
            if category in category_crops:
                # Filter and rank crops for the specific category
                category_specific_crops = []
                for crop_info in base_knowledge["best_crops"]:
                    for category_crop in category_crops[category]:
                        if category_crop.lower() in crop_info["crop"].lower():
                            category_specific_crops.append(crop_info)
                            break
                
                # If we have category-specific crops, use them; otherwise use all
                if category_specific_crops:
                    return {
                        "wasteType": request.wasteType,
                        "wasteCategory": waste_type_key,
                        "cropCategory": request.cropCategory,
                        "topCrops": category_specific_crops[:5],
                        "analysis": f"Based on agricultural science, {request.wasteType} is particularly beneficial for {request.cropCategory} crops"
                    }
        
        # Return top 5 crops with explanations
        return {
            "wasteType": request.wasteType,
            "wasteCategory": waste_type_key,
            "topCrops": base_knowledge["best_crops"][:5],
            "analysis": f"Based on agricultural science, {request.wasteType} provides optimal nutrients for these crops"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CropWasteCompatibilityRequest(BaseModel):
    wasteType: str
    wasteDescription: str
    cropCategory: Optional[str] = None

crop_waste_knowledge = {
    "Cattle Manure": {
        "best_crops": [
            {"crop": "Rice", "reason": "Provides balanced NPK nutrients essential for rice growth, improves soil structure for better water retention in paddies"},
            {"crop": "Corn", "reason": "High potassium content supports strong stalk development and kernel production"},
            {"crop": "Wheat", "reason": "Excellent source of organic matter that enhances wheat's root development and grain quality"},
            {"crop": "Vegetables (leafy)", "reason": "Rich in micronutrients that promote vigorous leaf growth in vegetables like lettuce and spinach"},
            {"crop": "Sugarcane", "reason": "Provides steady release of nutrients throughout the long growing season, boosting sugar content"}
        ]
    },
    "Poultry Waste": {
        "best_crops": [
            {"crop": "Leafy Vegetables", "reason": "Very high nitrogen content promotes rapid leaf growth in lettuce, spinach, and cabbage"},
            {"crop": "Corn", "reason": "Quick-release nitrogen fuels early vegetative growth, leading to taller plants"},
            {"crop": "Broccoli", "reason": "High nitrogen supports development of large heads and abundant foliage"},
            {"crop": "Cabbage", "reason": "Promotes tight head formation and large outer leaves"},
            {"crop": "Cauliflower", "reason": "Essential for curd development and overall plant vigor"}
        ]
    },
    "Swine Waste": {
        "best_crops": [
            {"crop": "Root Crops", "reason": "High phosphorus content promotes excellent root development in carrots, radishes, and sweet potatoes"},
            {"crop": "Fruit Trees", "reason": "Balanced nutrients support flowering, fruit set, and sweet fruit development"},
            {"crop": "Tomatoes", "reason": "Phosphorus-rich composition enhances flowering and fruit production"},
            {"crop": "Peppers", "reason": "Supports abundant flowering and larger fruit development"},
            {"crop": "Eggplant", "reason": "Essential nutrients for fruit set and plant vigor"}
        ]
    },
    "Goat Manure": {
        "best_crops": [
            {"crop": "Vegetables (all types)", "reason": "Mild composition won't burn plants, perfect for direct application in vegetable gardens"},
            {"crop": "Herbs", "reason": "Gentle nutrient release ideal for sensitive herbs like basil and oregano"},
            {"crop": "Salad Greens", "reason": "Provides steady nutrients without overwhelming delicate greens"},
            {"crop": "Beans", "reason": "Moderate nitrogen levels support growth without inhibiting nitrogen-fixing bacteria"},
            {"crop": "Peas", "reason": "Balanced nutrients support pod development and plant health"}
        ]
    },
    "Sheep Manure": {
        "best_crops": [
            {"crop": "Berry Crops", "reason": "High phosphorus and potassium promote flowering and sweet fruit development"},
            {"crop": "Grapes", "reason": "Potassium-rich composition enhances grape sweetness and vine health"},
            {"crop": "Flowering Plants", "reason": "Promotes abundant blooms and strong stem development"},
            {"crop": "Orchard Fruits", "reason": "Supports fruit set and improves fruit quality"},
            {"crop": "Strawberries", "reason": "Enhances flower production and berry sweetness"}
        ]
    },
    "Rabbit Manure": {
        "best_crops": [
            {"crop": "Vegetable Gardens", "reason": "Cold manure can be applied directly without composting, perfect for intensive vegetable production"},
            {"crop": "Tomatoes", "reason": "High nitrogen supports vigorous growth and fruit production"},
            {"crop": "Peppers", "reason": "Promotes healthy foliage and abundant fruiting"},
            {"crop": "Cucumbers", "reason": "Quick nutrient release supports rapid vine growth"},
            {"crop": "Squash", "reason": "Essential nutrients for large fruit development"}
        ]
    }
}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
