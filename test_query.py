import requests
import json
import time

def test_query_endpoint():
    """Test the /api/query endpoint"""
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(10)
    
    # Test data
    query_data = {
        "query": "The property in dispute was under the occupation",
        "k": 3
    }
    
    try:
        # Test the query endpoint
        print("🔍 Testing /api/query endpoint...")
        response = requests.post(
            "http://localhost:8000/api/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Query endpoint working!")
        else:
            print("❌ Query endpoint failed!")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on port 8000.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_upload_first():
    """First upload a document to create the FAISS index"""
    
    print("📄 Testing /api/upload first to create FAISS index...")
    
    # Create a simple test document
    test_content = """
    This is a legal document about property disputes.
    The property in question was under the occupation of the tenant.
    The landlord claims the tenant has violated the lease agreement.
    The dispute involves rental payments and property maintenance.
    """
    
    try:
        # Test the upload endpoint with text
        response = requests.post(
            "http://localhost:8000/process-text",
            json={"text": test_content, "filename": "test_document.txt"},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Upload Status: {response.status_code}")
        print(f"Upload Response: {response.json()}")
        
    except Exception as e:
        print(f"❌ Upload error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Microservice Endpoints")
    print("=" * 40)
    
    # First test upload
    test_upload_first()
    
    # Then test query
    test_query_endpoint()
