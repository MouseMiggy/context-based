# AgriLink Semantic Search Backend

FastAPI backend providing MPNet-based semantic search for livestock listings.

## Setup Instructions

### 1. Environment Setup

```bash
# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file in the backend directory:

```env
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/serviceAccount.json
FIRESTORE_PROJECT_ID=your-project-id
PORT=8000
```

### 3. Initial Data Ingestion

Run once to generate embeddings for existing listings:

```bash
python ingest_listings.py
```

### 4. Start the Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### POST `/embed`
Convert text to MPNet embedding.

**Request:**
```json
{
  "text": "healthy dairy cows for sale"
}
```

**Response:**
```json
{
  "embedding": [0.1234, 0.5678, ...]
}
```

### POST `/search`
Search livestock listings by semantic similarity.

**Request:**
```json
{
  "text": "dairy cows near manila",
  "top_k": 20
}
```

**Response:**
```json
{
  "matches": [
    {
      "id": "listing123",
      "score": 0.89,
      "data": { ...listing data... }
    }
  ]
}
```

### POST `/embed-listing`
Generate and store embedding for a specific listing.

**Request:**
```json
{
  "listingId": "listing123"
}
```

**Response:**
```json
{
  "message": "Embedding generated and stored successfully",
  "listingId": "listing123"
}
```

## Integration with Frontend

The frontend uses the `NEXT_PUBLIC_SEMANTIC_SEARCH_URL` environment variable to connect to this backend. Update your frontend `.env.local`:

```env
NEXT_PUBLIC_SEMANTIC_SEARCH_URL=http://localhost:8000
```

For production, replace with your deployed backend URL.

## Model Information

- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Embedding Dimension**: 768
- **Similarity Metric**: Cosine Similarity

## Semantic Text Construction

The system creates semantic text by combining:
- Listing name
- Description
- Category
- Owner location
- Breed (if available)
- Age (if available)

## Production Deployment

For production deployment:
1. Deploy this FastAPI service to a cloud provider (Render, Heroku, AWS, etc.)
2. Update `NEXT_PUBLIC_SEMANTIC_SEARCH_URL` in frontend to point to deployed backend
3. Ensure Firestore credentials are properly configured
4. Run initial ingestion script on production data

## Monitoring

- Check server logs for embedding generation status
- Monitor API response times
- Track semantic search usage and accuracy
