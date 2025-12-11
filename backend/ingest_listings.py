import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import firestore
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

# Initialize multilingual MPNet model
print("Loading multilingual MPNet model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
print("Multilingual model loaded successfully!")

# Initialize Firestore
service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
project_id = os.getenv('FIRESTORE_PROJECT_ID')

if service_account_path and os.path.exists(service_account_path):
    credentials = service_account.Credentials.from_service_account_file(service_account_path)
    db = firestore.Client(credentials=credentials, project=project_id)
else:
    db = firestore.Client(project=project_id)

def create_semantic_text(listing_data):
    """Create semantic search text from listing name only with Tagalog-English mappings"""
    parts = []
    
    # Add name only (not description/details as requested)
    if 'name' in listing_data:
        name = listing_data['name']
        parts.append(name)
        
        # Add Tagalog-English context for livestock terms
        tagalog_mappings = {
            'manok': 'chicken poultry',
            'baka': 'cattle cow beef', 
            'kalabaw': 'water buffalo carabao',
            'kabaw': 'water buffalo carabao',
            'kanding': 'goat',
            'kambing': 'goat',
            'itlog': 'egg poultry duck itik pugo pato',
            'pugo': 'quail duck',
            'pato': 'duck',
            'kuneho': 'rabbit',
            'baboy': 'pig swine boar'
        }
        
        # Check if name contains Tagalog terms and add English equivalents
        name_lower = name.lower()
        for tagalog, english in tagalog_mappings.items():
            if tagalog in name_lower:
                parts.append(english)
                break
    
    return ' '.join(parts)

def ingest_listings():
    """Generate and store MPNet embeddings for all livestock listings"""
    try:
        print("Starting ingestion process...")
        
        # Get all listings from Firestore
        listings_ref = db.collection('livestock_listings')
        docs = listings_ref.stream()
        
        # Convert to list and process in batches
        all_docs = list(docs)
        total_listings = len(all_docs)
        print(f"Found {total_listings} listings to process")
        
        processed_count = 0
        skipped_count = 0
        batch_size = 50  # Process 50 listings at a time
        
        for batch_start in range(0, total_listings, batch_size):
            batch_end = min(batch_start + batch_size, total_listings)
            batch_docs = all_docs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_listings-1)//batch_size + 1} (listings {batch_start+1}-{batch_end})")
            
            for doc in batch_docs:
                listing_data = doc.to_dict()
                listing_id = doc.id
                
                # Check if embedding already exists (force regenerate for multilingual model)
                if 'mpnet_embedding' in listing_data:
                    print(f"Updating {listing_id} - regenerating with multilingual model")
                else:
                    print(f"Processing {listing_id} - new embedding")
                
                # Create semantic text
                semantic_text = create_semantic_text(listing_data)
                
                if not semantic_text.strip():
                    print(f"Skipping {listing_id} - no semantic text available")
                    skipped_count += 1
                    continue
                
                # Generate embedding
                embedding = model.encode(semantic_text, convert_to_numpy=True)
                
                # Update document with embedding
                doc.reference.update({
                    'mpnet_embedding': embedding.tolist()
                })
                
                processed_count += 1
                print(f"Processed {doc.id} - embedding stored")
            
            # Progress indicator
            print(f"Batch complete. Total processed: {processed_count}/{total_listings}")
            
            # Small delay between batches to avoid rate limiting
            import time
            time.sleep(1)
        
        print(f"\nIngestion complete!")
        print(f"Processed: {processed_count} listings")
        print(f"Skipped: {skipped_count} listings")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        print(f"Progress so far: {processed_count} listings processed")
        raise

if __name__ == "__main__":
    ingest_listings()
