import os
import openai
import requests
import json
 
openai.api_key = "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"



def callGPT():

    system_prompt = "You will always describe the function call you're about to call before calling it."

    # user_content = input("What's up?\n")
    user_content = "What's the weather in Collingswood, NJ?"
    
    result = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        # stream = True, # Add this optional property.
        functions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ]
    )

    return result
 

def get_embedding(input):

    # Endpoint URL
    url = "https://lgretail2.openai.azure.com/openai/deployments/AdaEmbedding002/embeddings?api-version=2023-03-15-preview"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "api-key": "216ce4fb4d4c475b89d2c2c13fa5bd25"
    }
    
    # Body
    payload = {
        "input": input
    }
    
    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)
    
    # Extract the embedding vector from the response (assuming the response contains 'vector' key)
    embedding_vector = response.json()['data'][0]['embedding']
    
    return embedding_vector



# Function to query Pinecone DB
def query_pinecone_db(query_vector, top_k=7, namespace="Laptops"):
    # Endpoint URL
    url = "https://lg-pim-dev-d6c0941.svc.us-west4-gcp-free.pinecone.io/query"
    
    # Parameters
    payload = {
        "vector": [query_vector],
        "topK": top_k,
        "filter": {"Battery": "80Wh"},
        "includeMetadata": True,
        "includeValues": False,
        "namespace": namespace
    }
    
    headers = {
        "Content-Type": "application/json",
        "Api-Key": "ae2c9a79-7351-4615-a187-ba292601824f"
    }
    
    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)
    
    # Return the response (can be parsed as needed)
    return response.json()




# MAIN

# result = callGPT()
# print (result)
 
# for chunk in result:
#     x = chunk.choices[0]
#     print(x)



# Generate utterance embedding vector
embedding = get_embedding("can you show me laptops with 512 SSD memory")

# Get product results from Pinecone
results = query_pinecone_db(embedding)
print(json.dumps(results, indent=4))

 
print()