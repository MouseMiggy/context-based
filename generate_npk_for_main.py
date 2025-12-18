# Generate INDIVIDUAL_CROP_NPK for main.py

import hashlib

# Base NPK preferences from categories
BASE_CATEGORY_NPK = {
    'rice': {"n": "moderate", "p": "moderate", "k": "moderate", "organic_matter": "moderate"},
    'corn': {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "moderate"},
    'vegetables': {"n": "moderate", "p": "moderate", "k": "moderate", "organic_matter": "high"},
    'fruits': {"n": "moderate", "p": "moderate", "k": "high", "organic_matter": "high"},
    'rootCrops': {"n": "moderate", "p": "high", "k": "moderate", "organic_matter": "high"},
    'legumes': {"n": "low", "p": "moderate", "k": "moderate", "organic_matter": "high"},
    'herbs_spices': {"n": "moderate", "p": "moderate", "k": "low", "organic_matter": "moderate"},
    'industrial': {"n": "moderate", "p": "moderate", "k": "moderate", "organic_matter": "moderate"},
    'mushrooms': {"n": "low", "p": "moderate", "k": "moderate", "organic_matter": "very_high"}
}

# Crop-specific notes based on agricultural knowledge
CROP_NOTES = {
    # Rice variations
    'white-rice': 'Standard staple rice with balanced nutrient needs',
    'brown-rice': 'Higher nutrient requirements for bran development',
    'red-rice': 'Anthocyanin-rich variety needs moderate nutrients',
    'black-rice': 'Antioxidant-rich variety with moderate needs',
    'purple-rice': 'Similar to black rice, moderate requirements',
    'glutinous-rice': 'Sticky rice variety needs balanced NPK',
    'aromatic-rice': 'Fragrant varieties need moderate nutrients',
    'organic-rice': 'Grown without synthetic fertilizers',
    
    # Corn variations
    'yellow-corn': 'Standard field corn with balanced needs',
    'white-corn': 'Sweet white corn needs balanced nutrients',
    'sweet-corn': 'Higher sugar content needs moderate N',
    'popcorn': 'Hard kernel variety needs moderate nutrients',
    'feed-corn': 'High yield variety needs more nutrients',
    'hybrid-corn': 'Hybrid varieties need balanced NPK',
    'baby-corn': 'Young harvest needs moderate nutrients',
    
    # Leafy vegetables
    'spinach': 'Very high nitrogen for rapid leaf growth',
    'lettuce': 'High nitrogen for tender leaves',
    'kale': 'Nutrient-dense green needs high nitrogen',
    'cabbage': 'Head formation needs high phosphorus',
    'bok-choy': 'Fast-growing needs high nitrogen',
    'water-spinach': 'Aquatic vegetable needs high nitrogen',
    'moringa-leaves': 'Superfood needs moderate nitrogen',
    
    # Fruiting vegetables
    'tomato': 'High phosphorus and potassium for fruits',
    'eggplant': 'Needs balanced NPK for fruit development',
    'bitter-gourd': 'Tropical vine needs moderate nutrients',
    'squash': 'Heavy feeder needs high nutrients',
    'cucumber': 'High water and nutrient needs',
    'bell-pepper': 'Needs high potassium for fruit quality',
    'chili-pepper': 'Hot peppers need high potassium',
    
    # Root vegetables
    'carrot': 'Root development needs high phosphorus',
    'radish': 'Fast-growing needs moderate nutrients',
    'beetroot': 'Root and leaf both edible',
    'turnip': 'Cold crop needs balanced nutrients',
    'sweet-potato-root': 'High energy needs moderate NPK',
    'cassava-root': 'Starchy root needs low nitrogen',
    'taro-root': 'Tropical root needs high organic matter',
    'purple-yam-root': 'High value crop needs balanced NPK',
    
    # Fruits
    'mango': 'High potassium for fruit sweetness',
    'banana': 'Very high potassium for bunch development',
    'coconut': 'Very high potassium for oil production',
    'jackfruit': 'High nitrogen for large fruits',
    'durian': 'High NPK for premium quality',
    'papaya': 'Continuous fruiting needs high P and K',
    'avocado': 'High potassium for oil-rich fruit',
    'calamansi': 'High phosphorus for flowering',
    'watermelon': 'High water and potassium needs',
    
    # Legumes
    'mung-bean': 'Low nitrogen, fixes own N',
    'soybean': 'Low N needs, high P for protein',
    'peanut': 'High phosphorus for nut development',
    'green-peas': 'Low N, moderate P and K',
    'cowpea': 'Drought-tolerant needs balanced NPK',
    'string-bean': 'Moderate N needs for vine growth',
    
    # Herbs and spices
    'garlic-spice': 'Bulb development needs high phosphorus',
    'onion-spice': 'High phosphorus for bulb formation',
    'ginger-spice': 'Rhizome needs moderate nutrients',
    'turmeric-spice': 'High curcumin needs moderate NPK',
    'black-pepper': 'Vine needs balanced nutrients',
    'chili-spice': 'Hot spice needs high potassium',
    'basil-herb': 'Aromatic herb needs moderate N',
    'oregano-herb': 'Mediterranean herb needs low N',
    'mint-herb': 'Invasive herb needs moderate N',
    
    # Industrial crops
    'coffee': 'High quality beans need balanced NPK',
    'cacao': 'High potassium for pod development',
    'sugarcane': 'High yield needs high nitrogen',
    'tobacco': 'High nitrogen for leaf quality',
    'cotton': 'High potassium for boll development',
    'rubber': 'Moderate NPK for latex production',
    'abaca': 'Fiber crop needs moderate nutrients',
    'oil-palm': 'High potassium for oil production',
    
    # Mushrooms
    'oyster-mushroom': 'High organic matter needs',
    'button-mushroom': 'Compost-based needs high OM',
    'shiitake': 'Wood-grown needs specific nutrients',
    'straw-mushroom': 'Agricultural waste substrate',
    'enoki': 'Cold variety needs high organic matter'
}

def adjust_npk_level(base_level, variation):
    """Adjust NPK level based on variation"""
    levels = ['very_low', 'low', 'moderate', 'high', 'very_high']
    
    # Convert to index
    level_map = {'very_low': 0, 'low': 1, 'moderate': 2, 'high': 3, 'very_high': 4}
    base_idx = level_map.get(base_level, 2)
    
    # Apply variation with bounds checking
    new_idx = max(0, min(4, base_idx + variation))
    
    return levels[new_idx]

def generate_crop_variations():
    """Generate individual crop NPK preferences with deterministic variations"""
    INDIVIDUAL_CROP_NPK = {}
    
    # All crop IDs from the frontend
    all_crops = {
        'rice': [
            'white-rice', 'brown-rice', 'red-rice', 'black-rice', 'purple-rice',
            'glutinous-rice', 'aromatic-rice', 'lowland-rice', 'upland-rice',
            'heirloom-rice', 'organic-rice'
        ],
        'corn': [
            'yellow-corn', 'white-corn', 'sweet-corn', 'glutinous-corn',
            'popcorn', 'feed-corn', 'hybrid-corn', 'native-corn', 'baby-corn'
        ],
        'vegetables': [
            'bok-choy', 'mustard-greens', 'lettuce', 'spinach', 'water-spinach',
            'moringa-leaves', 'malabar-spinach', 'jute-leaves', 'cabbage',
            'chinese-cabbage', 'napa-cabbage', 'kale', 'swiss-chard', 'arugula',
            'sorrel', 'endive', 'tomato', 'eggplant', 'okra', 'bitter-gourd',
            'squash', 'cucumber', 'bell-pepper', 'chili-pepper', 'chayote',
            'bottle-gourd', 'sponge-gourd', 'ridge-gourd', 'winged-bean',
            'hyacinth-bean', 'yardlong-bean', 'snow-peas', 'green-peas',
            'zucchini', 'carrot', 'radish', 'beetroot', 'turnip', 'parsnip',
            'potato', 'sweet-potato', 'cassava', 'taro', 'purple-yam',
            'arrowroot', 'yam-bean', 'onion', 'garlic', 'leek', 'shallot',
            'asparagus', 'bamboo-shoots', 'celery', 'kohlrabi', 'cauliflower',
            'broccoli', 'banana-blossom', 'squash-flower', 'artichoke',
            'seaweed', 'sea-grapes', 'agar-seaweed', 'eucheuma', 'pako',
            'katuray-flower', 'talinum'
        ],
        'fruits': [
            'banana', 'mango', 'pineapple', 'papaya', 'coconut', 'jackfruit',
            'durian', 'rambutan', 'lanzones', 'mangosteen', 'guava', 'avocado',
            'calamansi', 'pomelo', 'orange', 'lemon', 'lime', 'watermelon',
            'melon', 'dragon-fruit', 'star-apple', 'sugar-apple', 'soursop',
            'santol', 'tamarind', 'passion-fruit', 'chico', 'duhat',
            'balimbing', 'bignay', 'macopa', 'longan', 'lychee', 'kiat-kiat',
            'breadfruit', 'marang', 'pili-nut-fruit', 'bael-fruit', 'kamias',
            'tamarillo', 'mulberry', 'strawberry', 'persimmon', 'fig', 'pear',
            'apple', 'plum', 'peach', 'cherry', 'blueberry', 'grapes'
        ],
        'rootCrops': [
            'sweet-potato-root', 'cassava-root', 'taro-root', 'purple-yam-root',
            'potato-root', 'arrowroot-root', 'yam-bean-root', 'radish-root',
            'carrot-root', 'beetroot-root', 'turnip-root', 'parsnip-root',
            'ginger-root', 'turmeric-root', 'galangal-root', 'lotus-root',
            'greater-yam', 'lesser-yam', 'elephant-foot-yam', 'purple-sweet-potato',
            'tapioca-root', 'jerusalem-artichoke', 'kudzu-root'
        ],
        'legumes': [
            'mung-bean', 'soybean', 'peanut', 'cowpea', 'string-bean',
            'winged-bean', 'hyacinth-bean', 'lima-bean', 'chickpea',
            'pigeon-pea', 'lentil', 'black-bean', 'red-kidney-bean',
            'white-bean', 'green-peas', 'snow-peas', 'split-peas',
            'fava-bean', 'adzuki-bean', 'navy-bean', 'pinto-bean',
            'jack-bean', 'sword-bean', 'velvet-bean', 'rice-bean',
            'bambara-groundnut', 'horse-gram'
        ],
        'herbs_spices': [
            'garlic-spice', 'onion-spice', 'shallot-spice', 'ginger-spice',
            'turmeric-spice', 'galangal-spice', 'black-pepper', 'white-pepper',
            'chili-spice', 'birds-eye-chili', 'paprika', 'cinnamon', 'cloves',
            'star-anise', 'nutmeg', 'mace', 'coriander-seed', 'cumin',
            'fennel', 'fenugreek', 'mustard-seed', 'allspice', 'bay-leaf',
            'vanilla', 'tamarind-spice', 'annatto', 'lemongrass-spice',
            'pandan-spice', 'kaffir-lime-leaf', 'curry-leaf', 'sesame-seed',
            'poppy-seed', 'cardamom', 'anise-seed', 'saffron', 'horseradish',
            'basil-herb', 'oregano-herb', 'thyme-herb', 'rosemary-herb',
            'mint-herb', 'lemongrass-herb', 'sambong', 'lagundi',
            'tsaang-gubat', 'akapulko', 'pandan-herb', 'ginger-herb',
            'turmeric-herb', 'garlic-herb', 'onion-herb', 'holy-basil',
            'peppermint', 'stevia', 'catnip', 'feverfew', 'gotu-kola',
            'alagaw', 'banaba', 'bitter-melon-leaves'
        ],
        'industrial': [
            'tobacco', 'rubber', 'abaca', 'cotton', 'coffee', 'cacao',
            'tea', 'hemp', 'oil-palm', 'sugarcane'
        ],
        'mushrooms': [
            'oyster-mushroom', 'button-mushroom', 'shiitake', 'straw-mushroom',
            'enoki', 'wood-ear', 'king-oyster', 'lions-mane', 'reishi',
            'maitake', 'porcini'
        ]
    }
    
    # Generate variations for each crop
    for category, crops in all_crops.items():
        base_npk = BASE_CATEGORY_NPK[category]
        
        for crop_id in crops:
            # Use hash for deterministic variation
            hash_obj = hashlib.md5(crop_id.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            
            # Create variation based on hash
            n_variation = (hash_int % 7) - 3  # -3 to 3
            p_variation = ((hash_int // 7) % 7) - 3
            k_variation = ((hash_int // 49) % 7) - 3
            
            # Apply variations
            n_level = adjust_npk_level(base_npk['n'], n_variation)
            p_level = adjust_npk_level(base_npk['p'], p_variation)
            k_level = adjust_npk_level(base_npk['k'], k_variation)
            om_level = base_npk['organic_matter']
            
            # Get specific note or generate generic one
            note = CROP_NOTES.get(crop_id, f"Optimized for {category.replace('_', ' ')} crops")
            
            INDIVIDUAL_CROP_NPK[crop_id] = {
                'n': n_level,
                'p': p_level,
                'k': k_level,
                'organic_matter': om_level,
                'note': note
            }
    
    return INDIVIDUAL_CROP_NPK

# Generate the dictionary
INDIVIDUAL_CROP_NPK = generate_crop_variations()

# Print as Python dictionary for main.py
print("INDIVIDUAL_CROP_NPK = {")
for crop, npk in INDIVIDUAL_CROP_NPK.items():
    print(f"    '{crop}': {{")
    print(f"        'n': '{npk['n']}',")
    print(f"        'p': '{npk['p']}',")
    print(f"        'k': '{npk['k']}',")
    print(f"        'organic_matter': '{npk['organic_matter']}',")
    print(f"        'note': '{npk['note']}'")
    print("    },")
print("}")
