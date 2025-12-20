# Semantic Embedding Generation Guide

## Overview

This document explains how AgriLink generates semantic embeddings for livestock waste listings to enable intelligent, context-aware search in the agricultural exchange platform.

---

## What is a Semantic Embedding?

A **semantic embedding** is a 768-dimensional vector (array of 768 numbers) that represents the **meaning** of text in a way that computers can understand and compare.

### Example:
```
Text: "Native Goat Manure for vegetables"
Embedding: [0.234, -0.567, 0.891, ..., 0.123] (768 numbers)
```

### Why 768 dimensions?
The MPNet model (paraphrase-multilingual-mpnet-base-v2) uses 768 dimensions to capture:
- Agricultural intent
- Livestock type
- Waste characteristics
- Crop suitability
- Language nuances (English/Tagalog/Cebuano)
- Farming context

---

## Embedding Generation Process

### 1. **Livestock Type as Context Anchor** (Most Reliable)

The livestock type dropdown is the most reliable field because:
- Users must select from predefined options
- No typos or inconsistencies
- Clear categorization

**Example:**
```python
livestock_types = ['goat']  # From dropdown
→ Normalized to: 'goats'
→ Context added: 'goat manure potassium rich vegetable fertilizer'
```

**Livestock Type Mappings:**
```python
{
    'chickens': 'poultry manure high nitrogen organic fertilizer',
    'cattle': 'cattle manure balanced nutrients soil improvement',
    'pigs': 'pig manure high phosphorus crop fertilizer',
    'goats': 'goat manure potassium rich vegetable fertilizer',
    'rabbits': 'rabbit manure gentle nutrients cold manure',
    'buffalo': 'buffalo manure organic matter paddy field fertilizer',
    'horses': 'horse manure mushroom substrate garden fertilizer',
    'sheep': 'sheep manure dry pellets organic fertilizer',
    'ducks': 'duck manure aquatic bird fertilizer'
}
```

### 2. **Waste Type Identification**

Automatically detects waste type from listing name:

```python
if 'manure' or 'dumi' in name:
    → 'animal manure livestock waste organic fertilizer'
    
if 'compost' or 'composted' in name:
    → 'composted organic matter soil amendment'
    
if 'egg' or 'itlog' or 'shell' in name:
    → 'eggshells calcium rich poultry waste'
    
if 'bedding' or 'litter' in name:
    → 'animal bedding livestock litter organic material'
```

### 3. **Agricultural Intent from Description**

Extracts farming-relevant information while **excluding**:
- ❌ Pricing (price, cost, pesos, ₱)
- ❌ Payment (cash, gcash, bank)
- ❌ Delivery (shipping, transport, pickup)
- ❌ Logistics (location, contact, quantity)

**Keeps only:**
- ✅ Crop types (rice, corn, vegetables)
- ✅ Soil benefits (nutrients, organic matter)
- ✅ Farming methods (organic, composted)
- ✅ Agricultural keywords (fertilizer, soil, plant)

**Example:**
```
Original: "Fresh goat manure. Good for vegetables. Price: 50 pesos per sack. 
           Available in Manila. Contact 09123456789."

Cleaned: "Fresh goat manure Good for vegetables"
```

### 4. **General Crop Suitability**

If no specific crops mentioned, adds general agricultural context:
```python
'suitable for crops vegetables rice corn general farming use'
```

### 5. **Organic Farming Context**

Always emphasizes organic and sustainable farming:
```python
'organic fertilizer sustainable agriculture soil health improvement'
```

### 6. **Multilingual Support (Tagalog-English)**

Automatically adds English equivalents for Tagalog terms:

```python
{
    'kambing/kanding': 'goat',
    'manok': 'chicken poultry',
    'baboy': 'pig swine',
    'baka': 'cattle cow',
    'kalabaw/kabaw': 'buffalo carabao',
    'kuneho': 'rabbit',
    'kabayo': 'horse',
    'tupa': 'sheep',
    'pato': 'duck'
}
```

---

## Complete Example

### Input Listing:
```json
{
  "name": "Native Goat Manure (Dumi ng Kambing katutubo)",
  "livestockTypes": ["goat"],
  "details": "Fresh organic goat manure. Rich in nutrients. Good for vegetables and rice. Price: 100 pesos per sack. Available in Quezon City. Contact me for delivery."
}
```

### Step-by-Step Processing:

**Step 1: Livestock Type Context**
```
'goats' + 'goat manure potassium rich vegetable fertilizer'
```

**Step 2: Waste Type Identification**
```
+ 'animal manure livestock waste organic fertilizer'
```

**Step 3: Clean Description**
```
Original: "Fresh organic goat manure. Rich in nutrients. Good for vegetables and rice. Price: 100 pesos per sack. Available in Quezon City. Contact me for delivery."

Cleaned: "Fresh organic goat manure Rich in nutrients Good for vegetables and rice"
(Removed: pricing, location, contact)
```

**Step 4: General Crop Suitability**
```
(Already has vegetables and rice, skip)
```

**Step 5: Organic Context**
```
+ 'organic fertilizer sustainable agriculture soil health improvement'
```

**Step 6: Tagalog-English Mapping**
```
+ 'goat' (from 'kambing' in name)
```

### Final Semantic Text:
```
goats goat manure potassium rich vegetable fertilizer animal livestock waste organic Fresh nutrients Good for vegetables and rice sustainable agriculture soil health improvement
```

### Generated Embedding:
```python
[0.234, -0.567, 0.891, 0.456, -0.123, ..., 0.789]  # 768 numbers
```

---

## What the Embedding Captures

For the example above, the 768-dimensional vector captures:

### 1. **Agricultural Intent** ✅
- Understands this is for farming/fertilizer use
- Not for selling livestock or other purposes

### 2. **Livestock Type** ✅
- Recognizes "goat" as the waste source
- Handles both "goat" and "kambing" (Tagalog)

### 3. **Intended Crop Use** ✅
- Knows it's good for vegetables and rice
- Can match queries like "fertilizer for vegetables"

### 4. **Quantity Context** ✅
- Understands "sack" as a unit (though not in final embedding)
- Focuses on agricultural use, not logistics

### 5. **Organic Farming Context** ✅
- Emphasizes organic and sustainable farming
- Matches queries about organic fertilizers

### 6. **Language Flexibility** ✅
- Handles English, Tagalog, Cebuano seamlessly
- "kambing" = "goat" = "kanding" (all match!)

---

## Similarity Matching

When a user searches for **"fertilizer for vegetables"**:

1. **Query Embedding Generated:**
   ```python
   query_embedding = model.encode("fertilizer for vegetables")
   # → [0.245, -0.556, 0.882, ...]
   ```

2. **Cosine Similarity Calculated:**
   ```python
   similarity = cosine_similarity(query_embedding, listing_embedding)
   # → 0.87 (87% match)
   ```

3. **Similarity Thresholds:**
   - **0.6 - 1.0** (60-100%): Excellent match ⭐⭐⭐
   - **0.5 - 0.6** (50-60%): Good match ⭐⭐
   - **0.4 - 0.5** (40-50%): Fair match ⭐
   - **< 0.4** (< 40%): Poor match ❌

### Why This Works:

**Query:** "fertilizer for vegetables"
**Listing:** "goat manure potassium rich vegetable fertilizer"

The embeddings are **close in vector space** because:
- Both mention "fertilizer"
- Both mention "vegetables"
- "goat manure" is semantically related to "fertilizer"
- "potassium rich" adds agricultural context

**Result:** High similarity score (0.87) → Top search result! 🎯

---

## API Usage

### Generate Embedding for Text

```bash
curl -X POST https://context-based-2.onrender.com/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "goat manure for vegetables"
  }'
```

**Response:**
```json
{
  "embedding": [0.234, -0.567, 0.891, ..., 0.123]
}
```

### Search Listings

```bash
curl -X POST https://context-based-2.onrender.com/search \
  -H "Content-Type: application/json" \
  -d '{
    "text": "fertilizer for vegetables",
    "top_k": 10
  }'
```

**Response:**
```json
{
  "matches": [
    {
      "id": "listing123",
      "score": 0.87,
      "data": { ...listing details... }
    }
  ]
}
```

### Generate Embedding for Specific Listing

```bash
curl -X POST https://context-based-2.onrender.com/embed-listing \
  -H "Content-Type: application/json" \
  -d '{
    "listingId": "listing123"
  }'
```

---

## Key Benefits

### 1. **Context-Aware Search** 🧠
- Understands intent, not just keywords
- "best for rice" matches listings with rice-friendly nutrients

### 2. **Multilingual Support** 🌐
- Handles English, Tagalog, Cebuano seamlessly
- "manok" = "chicken" = "poultry" (all match!)

### 3. **Agricultural Focus** 🌾
- Filters out pricing, logistics, contact info
- Focuses purely on farming relevance

### 4. **Reliable Context Anchor** ⚓
- Uses livestock type dropdown as foundation
- Reduces errors from user-entered text

### 5. **Synonym Matching** 🔄
- "fertilizer" matches "manure", "compost", "organic matter"
- "vegetables" matches "crops", "plants", "garden"

---

## Technical Details

### Model Information
- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Embedding Dimension**: 768
- **Similarity Metric**: Cosine Similarity
- **Languages**: 50+ languages (including English, Tagalog, Cebuano)

### Performance
- **Embedding Generation**: ~50-100ms per listing
- **Search Query**: ~150-200ms (400 listings)
- **Storage**: 3KB per embedding (768 floats × 4 bytes)

### Deployment
- **Platform**: Render.com
- **URL**: https://context-based-2.onrender.com
- **Status**: Always-on (Starter plan)
- **Region**: Singapore

---

## Maintenance

### Re-generate Embeddings

When listings are updated, re-generate embeddings:

```bash
# Local
python ingest_listings.py

# Or via API
curl -X POST https://context-based-2.onrender.com/embed-listing \
  -H "Content-Type: application/json" \
  -d '{"listingId": "listing123"}'
```

### Monitor Performance

Check logs for:
- Embedding generation time
- Search response time
- Similarity score distribution

---

## Summary

The semantic embedding system:

1. ✅ **Uses livestock type as reliable context anchor**
2. ✅ **Incorporates user description for agricultural intent**
3. ✅ **Focuses on waste source, crop usability, farming relevance**
4. ✅ **Excludes pricing, payment, delivery, logistics**
5. ✅ **Uses clear, neutral English**
6. ✅ **Assumes general agricultural suitability if details missing**
7. ✅ **Handles multilingual queries (English/Tagalog/Cebuano)**

**Result:** Intelligent, context-aware search that understands agricultural intent! 🎯🌾
