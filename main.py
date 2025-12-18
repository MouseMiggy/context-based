from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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

# Agricultural knowledge base with NPK values and crop requirements
CROP_CATEGORIES = {
    # Leafy vegetables (high nitrogen requirement)
    "leafy_vegetables": {
        "crops": ["spinach", "lettuce", "kale", "swiss-chard", "mustard-greens", "water-spinach", "moringa-leaves", "malabar-spinach", "jute-leaves", "arugula", "sorrel", "endive"],
        "npk_preference": {"n": "high", "p": "moderate", "k": "low"},
        "organic_matter": "high"
    },
    # Root vegetables (moderate nitrogen, high organic matter)
    "root_vegetables": {
        "crops": ["carrot", "radish", "beetroot", "turnip", "parsnip", "sweet-potato", "cassava", "taro", "purple-yam", "arrowroot", "yam-bean"],
        "npk_preference": {"n": "moderate", "p": "moderate", "k": "moderate"},
        "organic_matter": "high"
    },
    # Fruiting vegetables (balanced NPK)
    "fruiting_vegetables": {
        "crops": ["tomato", "eggplant", "bell-pepper", "chili-pepper", "cucumber", "squash", "chayote", "bottle-gourd", "sponge-gourd", "ridge-gourd", "zucchini", "pumpkin"],
        "npk_preference": {"n": "moderate", "p": "high", "k": "high"},
        "organic_matter": "moderate"
    },
    # Legumes (low nitrogen, high organic matter)
    "legumes": {
        "crops": ["green-peas", "snow-peas", "winged-bean", "hyacinth-bean", "yardlong-bean", "mung-bean", "soybean", "peanut"],
        "npk_preference": {"n": "low", "p": "moderate", "k": "moderate"},
        "organic_matter": "high"
    },
    # Brassica family (high nitrogen, moderate phosphorus)
    "brassica": {
        "crops": ["cabbage", "chinese-cabbage", "napa-cabbage", "bok-choy", "broccoli", "cauliflower", "kale"],
        "npk_preference": {"n": "high", "p": "moderate", "k": "moderate"},
        "organic_matter": "high"
    },
    # Allium family (moderate nitrogen)
    "allium": {
        "crops": ["onion", "garlic", "leek", "shallot"],
        "npk_preference": {"n": "moderate", "p": "moderate", "k": "moderate"},
        "organic_matter": "moderate"
    },
    # Fruits (balanced NPK)
    "fruits": {
        "crops": ["banana", "mango", "pineapple", "papaya", "coconut", "jackfruit", "durian", "rambutan", "lanzones", "mangosteen", "guava", "avocado", "calamansi", "pomelo", "orange", "lemon", "lime", "watermelon", "melon", "dragon-fruit", "strawberry", "grapes"],
        "npk_preference": {"n": "moderate", "p": "moderate", "k": "high"},
        "organic_matter": "moderate"
    },
    # Cereals (moderate N and P)
    "cereals": {
        "crops": ["white-rice", "brown-rice", "red-rice", "black-rice", "purple-rice", "glutinous-rice", "aromatic-rice", "lowland-rice", "upland-rice", "yellow-corn", "white-corn", "sweet-corn", "wheat", "sorghum", "millet"],
        "npk_preference": {"n": "moderate", "p": "moderate", "k": "moderate"},
        "organic_matter": "moderate"
    },
    # Herbs (moderate nitrogen)
    "herbs": {
        "crops": ["basil", "oregano", "thyme", "rosemary", "parsley", "cilantro", "dill", "mint"],
        "npk_preference": {"n": "moderate", "p": "moderate", "k": "low"},
        "organic_matter": "moderate"
    }
}

# Livestock waste NPK database (average values) with aliases
WASTE_ALIASES = {
    "cattle": ["cattle", "cow", "bull", "carabao", "kalabaw"],
    "chicken": ["chicken", "poultry", "hen", "rooster", "pugo"],
    "pig": ["pig", "swine", "hog", "baboy"],
    "goat": ["goat", "kanding"],
    "sheep": ["sheep"],
    "horse": ["horse", "equine"],
    "rabbit": ["rabbit", "kuneho"],
    "duck": ["duck", "itik", "pato"],
    "quail": ["quail"]
}

def detect_waste_type(waste_name):
    """Detect waste type from name using aliases"""
    waste_name = waste_name.lower()
    
    for waste_type, aliases in WASTE_ALIASES.items():
        for alias in aliases:
            if alias in waste_name:
                return waste_type
    
    return None

WASTE_NPK = {
    "cattle": {
        "n": 2.5,  # %
        "p": 1.2,
        "k": 1.8,
        "organic_matter": 85,  # %
        "best_for": ["leafy_vegetables", "cereals", "brassica"],
        "notes": "Well-balanced, excellent for soil structure. Best composted.",
        "usage": "Compost for 2-3 months before application"
    },
    "chicken": {
        "n": 3.0,
        "p": 2.5,
        "k": 1.8,
        "organic_matter": 75,
        "best_for": ["leafy_vegetables", "fruiting_vegetables", "cereals"],
        "notes": "High in nitrogen and phosphorus. Very concentrated.",
        "usage": "Must be composted or aged 3-6 months. Use at half rate of other manures."
    },
    "pig": {
        "n": 2.0,
        "p": 2.0,
        "k": 1.5,
        "organic_matter": 70,
        "best_for": ["root_vegetables", "fruiting_vegetables", "fruits"],
        "notes": "High phosphorus content. Good for root development.",
        "usage": "Compost thoroughly. Good for fruit and root crops."
    },
    "goat": {
        "n": 2.2,
        "p": 1.5,
        "k": 2.0,
        "organic_matter": 80,
        "best_for": ["leafy_vegetables", "root_vegetables", "herbs"],
        "notes": "Mild and less odorous. Good all-purpose manure.",
        "usage": "Can be used fresh for established plants, compost for seedlings."
    },
    "sheep": {
        "n": 2.0,
        "p": 1.8,
        "k": 2.5,
        "organic_matter": 75,
        "best_for": ["fruits", "cereals", "brassica"],
        "notes": "Higher in potassium. Good for fruit quality.",
        "usage": "Excellent for orchards and fruit trees."
    },
    "horse": {
        "n": 1.5,
        "p": 1.0,
        "k": 1.5,
        "organic_matter": 85,
        "best_for": ["root_vegetables", "legumes", "herbs"],
        "notes": "Lower nitrogen, high organic matter. Cool manure.",
        "usage": "Can be used without composting on established plants."
    },
    "rabbit": {
        "n": 2.4,
        "p": 1.4,
        "k": 0.6,
        "organic_matter": 80,
        "best_for": ["leafy_vegetables", "vegetables"],
        "notes": "Cold manure, can be applied directly.",
        "usage": "Perfect for intensive vegetable production."
    },
    "duck": {
        "n": 2.8,
        "p": 2.0,
        "k": 1.2,
        "organic_matter": 70,
        "best_for": ["leafy_vegetables", "aquatic_crops"],
        "notes": "Higher nitrogen than chicken. Wet consistency.",
        "usage": "Dry and compost before use. Excellent for rice paddies."
    },
    "quail": {
        "n": 3.5,
        "p": 2.8,
        "k": 2.0,
        "organic_matter": 65,
        "best_for": ["leafy_vegetables", "fruiting_vegetables"],
        "notes": "Very concentrated. Highest nitrogen content.",
        "usage": "Use sparingly. Must be well-composted."
    }
}

# Individual crop NPK preferences for more granular scoring
INDIVIDUAL_CROP_NPK = {
    # Fruits with specific needs
    "mango": {"n": "moderate", "p": "moderate", "k": "high", "organic_matter": "moderate", "note": "High potassium for fruit sweetness and development"},
    "coconut": {"n": "moderate", "p": "moderate", "k": "very_high", "organic_matter": "moderate", "note": "Very high potassium for oil production and husk development"},
    "jackfruit": {"n": "high", "p": "moderate", "k": "high", "organic_matter": "moderate", "note": "High nitrogen for rapid growth of large fruits"},
    "durian": {"n": "high", "p": "high", "k": "high", "organic_matter": "high", "note": "High NPK for premium fruit quality and strong aroma"},
    "banana": {"n": "high", "p": "high", "k": "very_high", "organic_matter": "high", "note": "Very high potassium for bunch development and sweetness"},
    "papaya": {"n": "moderate", "p": "high", "k": "high", "organic_matter": "moderate", "note": "High phosphorus for continuous fruiting"},
    "avocado": {"n": "moderate", "p": "moderate", "k": "high", "organic_matter": "high", "note": "High potassium for oil-rich fruit development"},
    "calamansi": {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "moderate", "note": "High phosphorus for flowering and fruit set"},
    
    # Vegetables with specific needs
    "spinach": {"n": "very_high", "p": "moderate", "k": "low", "organic_matter": "high", "note": "Very high nitrogen for rapid leaf growth"},
    "lettuce": {"n": "very_high", "p": "moderate", "k": "low", "organic_matter": "high", "note": "Very high nitrogen for tender leaves"},
    "tomato": {"n": "moderate", "p": "high", "k": "high", "organic_matter": "moderate", "note": "High phosphorus and potassium for fruit production"},
    "eggplant": {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "moderate", "note": "High phosphorus for fruit development"},
    "chili-pepper": {"n": "moderate", "p": "high", "k": "high", "organic_matter": "moderate", "note": "High potassium for capsaicin production"},
    "carrot": {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "high", "note": "High phosphorus for root development"},
    "radish": {"n": "moderate", "p": "moderate", "k": "moderate", "organic_matter": "high", "note": "Balanced NPK for quick root development"},
    "onion": {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "moderate", "note": "High phosphorus for bulb formation"},
    "garlic": {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "moderate", "note": "High phosphorus for clove development"},
    
    # Cereals with specific needs
    "white-rice": {"n": "moderate", "p": "moderate", "k": "moderate", "organic_matter": "moderate", "note": "Balanced NPK for grain production"},
    "yellow-corn": {"n": "moderate", "p": "high", "k": "high", "organic_matter": "moderate", "note": "High phosphorus and potassium for ear development"},
    "sweet-corn": {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "moderate", "note": "High phosphorus for sweet kernel development"},
    
    # Legumes with specific needs
    "green-peas": {"n": "low", "p": "moderate", "k": "moderate", "organic_matter": "high", "note": "Low nitrogen as they fix their own"},
    "mung-bean": {"n": "low", "p": "moderate", "k": "moderate", "organic_matter": "high", "note": "Low nitrogen, high organic matter for pod development"},
    "peanut": {"n": "low", "p": "high", "k": "moderate", "organic_matter": "high", "note": "High phosphorus for nut development"},
    
    # Herbs with specific needs
    "basil": {"n": "moderate", "p": "moderate", "k": "low", "organic_matter": "moderate", "note": "Moderate nitrogen for aromatic leaf production"},
    "oregano": {"n": "moderate", "p": "moderate", "k": "low", "organic_matter": "moderate", "note": "Low potassium for essential oil concentration"},
    "mint": {"n": "moderate", "p": "moderate", "k": "moderate", "organic_matter": "moderate", "note": "Balanced NPK for vigorous leaf growth"}
}

def get_crop_category(crop_id):
    """Map crop ID to its agricultural category"""
    crop_id = crop_id.lower().replace('-', ' ').replace('_', ' ')
    
    for category, data in CROP_CATEGORIES.items():
        if crop_id in [c.lower().replace('-', ' ').replace('_', ' ') for c in data["crops"]]:
            return category
    
    # Fallback: try to match by keywords
    if any(word in crop_id for word in ["rice", "corn", "wheat", "sorghum", "millet"]):
        return "cereals"
    elif any(word in crop_id for word in ["tomato", "pepper", "eggplant", "cucumber", "squash"]):
        return "fruiting_vegetables"
    elif any(word in crop_id for word in ["spinach", "lettuce", "kale", "greens"]):
        return "leafy_vegetables"
    elif any(word in crop_id for word in ["carrot", "radish", "beet", "potato", "turnip"]):
        return "root_vegetables"
    elif any(word in crop_id for word in ["pea", "bean", "soy", "peanut"]):
        return "legumes"
    elif any(word in crop_id for word in ["cabbage", "broccoli", "cauliflower"]):
        return "brassica"
    elif any(word in crop_id for word in ["onion", "garlic", "leek"]):
        return "allium"
    elif any(word in crop_id for word in ["mango", "banana", "apple", "orange", "grape"]):
        return "fruits"
    elif any(word in crop_id for word in ["basil", "oregano", "thyme", "mint"]):
        return "herbs"
    
    return "general"  # Default category

def calculate_compatibility_score(waste_type, crop_category, crop_id=None):
    """Calculate compatibility score based on NPK matching and crop requirements"""
    if crop_category not in CROP_CATEGORIES or waste_type not in WASTE_NPK:
        return 0
    
    waste = WASTE_NPK[waste_type]
    
    # Check if we have individual crop preferences
    crop_id_clean = crop_id.lower().replace('-', ' ').replace('_', ' ') if crop_id else None
    print(f"DEBUG: crop_id={crop_id}, crop_id_clean={crop_id_clean}")
    
    if crop_id_clean and crop_id_clean in INDIVIDUAL_CROP_NPK:
        crop_req = INDIVIDUAL_CROP_NPK[crop_id_clean]
        crop_note = crop_req.get('note', '')
        print(f"DEBUG: Using individual crop preferences for {crop_id_clean}")
    else:
        crop_req = CROP_CATEGORIES[crop_category]["npk_preference"]
        crop_note = f"Suitable for {crop_category.replace('_', ' ')}"
        print(f"DEBUG: Using category preferences for {crop_category}")
    
    score = 0
    max_score = 100
    
    # NPK matching score (70% of total)
    npk_score = 0
    
    # Nitrogen preference - handle very_high
    n_target = {
        "very_high": 3.0,
        "high": 2.5,
        "moderate": 1.5,
        "low": 1.0
    }.get(crop_req["n"], 1.5)
    
    if crop_req["n"] == "very_high" and waste["n"] >= 3.0:
        npk_score += 30
    elif crop_req["n"] == "high" and waste["n"] >= 2.5:
        npk_score += 30
    elif crop_req["n"] == "moderate" and 1.5 <= waste["n"] <= 2.5:
        npk_score += 30
    elif crop_req["n"] == "low" and waste["n"] <= 2.0:
        npk_score += 30
    else:
        # Partial match
        diff = abs(waste["n"] - n_target)
        npk_score += max(0, 30 - diff * 10)
    
    # Phosphorus preference
    p_target = {
        "very_high": 2.5,
        "high": 2.0,
        "moderate": 1.5,
        "low": 1.0
    }.get(crop_req["p"], 1.5)
    
    if crop_req["p"] == "very_high" and waste["p"] >= 2.5:
        npk_score += 20
    elif crop_req["p"] == "high" and waste["p"] >= 2.0:
        npk_score += 20
    elif crop_req["p"] == "moderate" and 1.0 <= waste["p"] <= 2.0:
        npk_score += 20
    elif crop_req["p"] == "low" and waste["p"] <= 1.5:
        npk_score += 20
    else:
        diff = abs(waste["p"] - p_target)
        npk_score += max(0, 20 - diff * 10)
    
    # Potassium preference - handle very_high
    k_target = {
        "very_high": 2.5,
        "high": 2.0,
        "moderate": 1.5,
        "low": 1.0
    }.get(crop_req["k"], 1.5)
    
    if crop_req["k"] == "very_high" and waste["k"] >= 2.5:
        npk_score += 20
    elif crop_req["k"] == "high" and waste["k"] >= 2.0:
        npk_score += 20
    elif crop_req["k"] == "moderate" and 1.0 <= waste["k"] <= 2.0:
        npk_score += 20
    elif crop_req["k"] == "low" and waste["k"] <= 1.5:
        npk_score += 20
    else:
        diff = abs(waste["k"] - k_target)
        npk_score += max(0, 20 - diff * 10)
    
    score += npk_score * 0.7
    
    # Organic matter preference (20% of total)
    om_target = {
        "very_high": 90,
        "high": 85,
        "moderate": 75,
        "low": 70
    }.get(crop_req.get("organic_matter", "moderate"), 75)
    
    if crop_req.get("organic_matter") == "very_high" and waste["organic_matter"] >= 90:
        score += 20
    elif crop_req.get("organic_matter") == "high" and waste["organic_matter"] >= 80:
        score += 20
    elif crop_req.get("organic_matter") == "moderate" and 70 <= waste["organic_matter"] <= 85:
        score += 20
    elif crop_req.get("organic_matter") == "low" and waste["organic_matter"] <= 75:
        score += 20
    else:
        diff = abs(waste["organic_matter"] - om_target)
        score += max(0, 20 - diff * 0.5)
    
    # Best for bonus (10% of total)
    if crop_category in waste["best_for"]:
        score += 10
    
    # Store the crop note for use in the response
    if not hasattr(calculate_compatibility_score, '_crop_notes'):
        calculate_compatibility_score._crop_notes = {}
    calculate_compatibility_score._crop_notes[crop_id_clean] = crop_note
    
    return min(max_score, score)

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

class BatchCropWasteRequest(BaseModel):
    wasteType: str
    wasteDescription: str
    cropCategory: Optional[str] = None
    listings: List[Dict[str, Any]]

class CropCompatibilityRequest(BaseModel):
    cropIds: List[str]  # List of specific crop IDs from user
    cropCategory: Optional[str] = None  # Optional crop category filter

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

class CropWasteCompatibilityRequest(BaseModel):
    wasteType: str
    wasteDescription: str
    cropCategory: Optional[str] = None

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

@app.post("/crop-compatibility-analysis")
async def analyze_crop_compatibility(request: CropCompatibilityRequest):
    """Analyze crop-waste compatibility using agricultural science rules"""
    try:
        # Get all listings from Firestore
        listings_ref = db.collection('livestock_listings')
        docs = listings_ref.stream()
        
        results = []
        
        for doc in docs:
            listing_data = doc.to_dict()
            listing_id = doc.id
            
            # Skip sold or deleted listings
            if listing_data.get('status') in ['sold', 'deleted']:
                continue
            
            # Extract waste type from listing name
            waste_name = listing_data.get('name', '').lower()
            waste_type = detect_waste_type(waste_name)
            
            # If no specific waste type found, skip
            if not waste_type:
                continue
            
            # Analyze compatibility with each user crop
            crop_scores = []
            for crop_id in request.cropIds:
                # Get crop category
                crop_category = get_crop_category(crop_id)
                
                # Calculate compatibility score using agricultural rules
                compatibility_score = calculate_compatibility_score(waste_type, crop_category, crop_id)
                
                # Get crop display name
                crop_name = crop_id.replace('-', ' ').replace('_', ' ').title()
                
                if compatibility_score > 30:  # Only include decent matches
                    crop_scores.append({
                        'cropId': crop_id,
                        'cropName': crop_name,
                        'score': compatibility_score,
                        'category': crop_category
                    })
            
            # Sort crops by score
            crop_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Only include if at least one crop has good compatibility (>40%)
            if crop_scores and crop_scores[0]['score'] > 40:
                # Create detailed top crops with NPK info
                top_crops = []
                for crop in crop_scores[:5]:
                    waste_info = WASTE_NPK[waste_type]
                    crop_id_clean = crop['cropId'].lower().replace('-', ' ').replace('_', ' ')
                    crop_note = INDIVIDUAL_CROP_NPK.get(crop_id_clean, {}).get('note', f"Suitable for {crop['category'].replace('_', ' ')}")
                    
                    top_crops.append({
                        'cropId': crop['cropId'],
                        'cropName': crop['cropName'],
                        'score': crop['score'],
                        'reason': f"{waste_type.title()} manure - N:{waste_info['n']}%, P:{waste_info['p']}%, K:{waste_info['k']}%, {waste_info['organic_matter']}% organic matter. {crop_note} {waste_info['notes']} Usage: {waste_info['usage']}",
                        'npk': {
                            'n': waste_info['n'],
                            'p': waste_info['p'],
                            'k': waste_info['k'],
                            'organic_matter': waste_info['organic_matter']
                        },
                        'usage': waste_info['usage']
                    })
                
                results.append({
                    'listingId': listing_id,
                    'listingData': listing_data,
                    'wasteType': waste_type,
                    'cropScores': top_crops
                })
        
        # Sort results by highest compatibility score
        results.sort(key=lambda x: x['cropScores'][0]['score'], reverse=True)
        
        return {
            'compatibleListings': results,
            'totalFound': len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-compatibility")
async def test_compatibility():
    """Test the compatibility system with example crops"""
    test_crops = ["spinach", "tomato", "carrot", "white-rice", "green-peas"]
    
    results = {}
    for crop_id in test_crops:
        crop_category = get_crop_category(crop_id)
        crop_scores = []
        
        for waste_type in WASTE_NPK.keys():
            score = calculate_compatibility_score(waste_type, crop_category)
            if score > 30:
                waste_info = WASTE_NPK[waste_type]
                crop_scores.append({
                    'wasteType': waste_type,
                    'score': score,
                    'reason': f"N:{waste_info['n']}%, P:{waste_info['p']}%, K:{waste_info['k']}%, {waste_info['organic_matter']}% organic matter. {waste_info['notes']}"
                })
        
        crop_scores.sort(key=lambda x: x['score'], reverse=True)
        results[crop_id] = {
            'category': crop_category,
            'top5': crop_scores[:5]
        }
    
    return results

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
