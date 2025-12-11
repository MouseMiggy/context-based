import requests
import json

def test_backend():
    """Test if the semantic search backend is running and working"""
    try:
        print("🔍 Testing backend connection...")
        
        # Test basic health
        response = requests.get("http://localhost:8000")
        print(f"📡 Backend health check: {response.status_code}")
        
        # Test semantic search
        search_data = {
            "text": "manok",
            "top_k": 5
        }
        
        response = requests.post("http://localhost:8000/search", 
                               json=search_data,
                               headers={"Content-Type": "application/json"})
        
        print(f"📥 Search response status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"📊 Found {len(result.get('matches', []))} matches")
            
            if result.get('matches'):
                print("🎯 Top matches:")
                for i, match in enumerate(result['matches'][:3], 1):
                    print(f"  {i}. ID: {match['id']}, Score: {match['score']:.4f}")
            else:
                print("❌ No matches found")
                
        else:
            print(f"❌ Search failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend - is it running?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_backend()
