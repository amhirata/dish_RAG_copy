import streamlit as st
import openai
import pinecone
from pinecone import ServerlessSpec

import firebase_admin
from firebase_admin import credentials, firestore
import time
import os

from dotenv import load_dotenv
load_dotenv()

firebase_credentials = json.loads(st.secrets["firebase_credentials"])
# -----------------------
# 1. INITIAL SETUP
# -----------------------

# Initialize Firebase Admin SDK
# Make sure you have your Firebase service account JSON in the same folder or specify the path
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
# Get Firestore client
db = firestore.client()

# Initialize Pinecone
pinecone_key = os.environ.get('pinecone_key')
pinecone_client = pinecone.Pinecone(api_key=pinecone_key)
index_name = "recipe-index"
index = pinecone_client.Index(index_name)


# Initialize OpenAI
openai_key = os.environ.get('openai_key')
# Instead of setting openai.api_key, create an OpenAI client instance:
client = openai.OpenAI(api_key=openai_key)


# -----------------------
# 2. HELPER FUNCTIONS
# -----------------------

## PINECONE: Querying the Index
# Define get recipe embedding function
def get_recipe_embedding(text):
    # Use the client to create embeddings:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # The API returns a list of embeddings; we take the first one
    embedding = response.data[0].embedding  # Access using .data[0].embedding
    return embedding

def extract_recipe_ids(data):
    # Extract recipe_ids from the metadata of each match
    recipe_ids = [match['metadata']['recipe_id'] for match in data['matches']]
    return recipe_ids

def get_relevant_recipes(user_query, max_time=30, top_k=5):
    """
    user_query: e.g. "vegan dish with chickpeas under 30 minutes, also baby-friendly"
    max_time: an integer from the slider
    top_k: number of recipes to retrieve
    """
    # 1. Identify filters
    is_vegan_filter = "vegan" in user_query.lower()
    is_baby_filter = "baby-friendly" in user_query.lower()
    is_vegetarian_filter = "vegetarian" in user_query.lower()
    is_gluten_free_filter = "gluten-free" in user_query.lower()
    is_dairy_free_filter = "dairy-free" in user_query.lower()
    is_nut_free_filter = "nut-free" in user_query.lower()

    # 2. Get embedding
    query_embedding = get_recipe_embedding(user_query)

    # 3. Build Pinecone filter
    pinecone_filter = {}

    # Time filter (only if max_time > 0)
    if max_time > 0:
        pinecone_filter["total_time_mins"] = {"$lte": max_time}

    # Dietary filters
    if is_vegan_filter:
        pinecone_filter["is_vegan"] = True
    if is_baby_filter:
        pinecone_filter["is_baby_friendly"] = True
    if is_vegetarian_filter:
        pinecone_filter["is_vegetarian"] = True
    if is_gluten_free_filter:
        pinecone_filter["is_gluten_free"] = True
    if is_dairy_free_filter:
        pinecone_filter["is_dairy_free"] = True
    if is_nut_free_filter:
        pinecone_filter["is_nut_free"] = True

    # 4. Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter if pinecone_filter else None
    )
    print("Pinecone query results:", results)

    # 5. Format results
    recipes = []
    if "matches" in results:
        for match in results['matches']:
            metadata = match['metadata']
            recipe_info = {
                'id': match['id'],
                'score': match['score'],
                'name': metadata.get('name', ''),
                'recipe_id': metadata.get('recipe_id'),
                'is_vegan': metadata.get('is_vegan'),
                'is_baby_friendly': metadata.get('is_baby_friendly'),
                'ratings': metadata.get('ratings','N/A'),
                'total_time_mins': metadata.get('total_time_mins','N/A')
            }
            recipes.append(recipe_info)

    return recipes

## FIRESTORE: Fetching a Recipe from Firestore
# Create a helper function to fetch recipe data by ID:
def get_recipe_from_db(doc_id):
    """
    Fetch the recipe document from Firestore by doc_id, retrieving only entries
    with 'embedding' field.
    Returns None if no document found.
    """
    doc_ref = db.collection("recipes").document(doc_id)
    doc = doc_ref.get()

    if doc.exists:
        data = doc.to_dict()
        # Remove embedding for brevity
        if 'embedding' in data:
            del data['embedding']
        return data

    return None

def build_recipe_text_blocks(retrieved_recipes):
    recipe_text_blocks = []

    for i, recipe in enumerate(retrieved_recipes, start=1):
        # This is the doc ID we got from Pinecone metadata
        doc_id = recipe['recipe_id']  
        
        # Fetch by the doc ID in Firestore
        full_recipe = get_recipe_from_db(doc_id)
        if not full_recipe:
            continue
        
        # Now read the fields from Firestore
        name = full_recipe.get("name", "Unknown Recipe")
        ingredients = full_recipe.get("ingredients", [])
        steps = full_recipe.get("steps", [])
        description = full_recipe.get("description", "")
        ratings = full_recipe.get("ratings", "N/A")
        vote_count = full_recipe.get("vote_count", "N/A")
        total_time_mins = full_recipe.get("total_time_mins", "N/A")
        serves = full_recipe.get("serves", "N/A")
        image = full_recipe.get("image", "N/A")
        recipe_type = full_recipe.get("subcategory", "N/A")
        difficulty = full_recipe.get("difficulty", "N/A")

        # The doc also has a field named "id" (often the same as doc_id)
        # If you really want to show that field, rename to avoid confusion:
        field_id = full_recipe.get("id", "N/A")

        recipe_url = full_recipe.get("url", "N/A")

        block = (
            f"{i}) {name}\n"
            f"Description: {description}\n"
            f"Ingredients: {', '.join(ingredients)}\n"
            f"Steps: {' '.join(steps)}\n"
            f"Ratings: {ratings}\n"
            f"Total Time: {total_time_mins} minutes\n"
            f"Firestore Doc ID: {doc_id}\n"
            f"Field 'id': {field_id}\n"
            f"Vote Count: {vote_count}\n"
            f"Serving Size: {serves}\n"
            f"Image: {image}\n"
            f"Recipe URL: {recipe_url}\n"
            f"Recipe Type: {recipe_type}\n"
            f"Difficulty: {difficulty}\n"
        )
        recipe_text_blocks.append(block)

    return recipe_text_blocks

def generate_recipe_suggestion_with_db(retrieved_recipes, user_query):
    """
    Takes top retrieved recipes (with minimal metadata + ID from Pinecone)
    plus user query, fetches full recipe data from Firestore,
    crafts a prompt, calls GPT-3.5, returns the model's response.
    """
    recipe_text_blocks = build_recipe_text_blocks(retrieved_recipes)
    combined_recipe_text = "\n".join(recipe_text_blocks)

    prompt = f"""
    You are a helpful chef.
    The user wants: {user_query}.
    Here are {len(retrieved_recipes)} recipe options that match:
    {combined_recipe_text}
    Using ONLY the above information, please suggest the best recipe or a new creative combination based on these options.
    DO NOT MAKE UP RECIPES.
    Reprint the name, description, author, average user ratings, number of votes, ingredients (including measurements), steps, serving size, difficulty, recipe type, total prep/cooking time and the URL of the suggested recipe.
    Explain why it's a good fit, and keep it concise.
    Only use the recipes provided for your answer.
    At the end print the name, description, author, average user ratings, recipe type, number of votes and the URL of two other top ranked recipes and ask if the user would like to see the full recipe of either.
    At the bottom of your response print the recipe IDs of each of the recipes referenced in your response.
    """
    ###
    print("DEBUG PROMPT:", prompt)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful recipe assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    return answer

# -----------------------
# 3. STREAMLIT APP
# -----------------------

def main():
    st.title("🍳 DishRAG: Recipe Search")

    # Initialize session_state for chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "conversation_active" not in st.session_state:
        st.session_state["conversation_active"] = False

    st.markdown("#### Enter your recipe criteria below:")

    # Input fields
    ingredients = st.text_input("Available Ingredients (comma-separated)",
                                placeholder="e.g. chicken, broccoli, cheddar")

    max_time = st.slider("Maximum Prep/Cook Time (minutes)", 
                         min_value=0, 
                         max_value=120, 
                         value=30, 
                         step=5)
    
    # Checkboxes for dietary restrictions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vegan = st.checkbox("Vegan")
        baby_friendly = st.checkbox("Baby-friendly")
    with col2:
        vegetarian = st.checkbox("Vegetarian")
    with col3:
        gluten_free = st.checkbox("Gluten-Free")
    with col4:
        dairy_free = st.checkbox("Dairy-Free")

    # Build a user query string from these inputs
    dietary_flags = []
    if vegan: 
        dietary_flags.append("vegan")
    if baby_friendly:
        dietary_flags.append("baby-friendly")
    if vegetarian:
        dietary_flags.append("vegetarian")
    if gluten_free:
        dietary_flags.append("gluten-free")
    if dairy_free:
        dietary_flags.append("dairy-free")

    user_query = ""
    if ingredients:
        user_query += f"Ingredients: {ingredients}. "
    if max_time > 0:
        user_query += f"Under {max_time} minutes. "
    if dietary_flags:
        user_query += "Dietary: " + ", ".join(dietary_flags) + ". "

    st.write("**Computed user query**:", user_query if user_query else "No query yet.")

    # Button to start the chat
    if st.button("Get Recipe Suggestions"):
        if not user_query.strip():
            st.warning("Please provide some criteria or ingredients.")
        else:
            # 1) Retrieve from Pinecone
            retrieved_recipes = get_relevant_recipes(user_query, max_time=max_time, top_k=3)

            # 2) Generate initial LLM suggestion
            llm_response = generate_recipe_suggestion_with_db(retrieved_recipes, user_query)
            
            # 3) Initialize conversation in session state
            st.session_state["messages"] = []
            st.session_state["messages"].append({"role": "assistant", "content": llm_response})
            st.session_state["conversation_active"] = True

            # 4) Display the retrieved recipes with expanders
            st.write("### Suggested Recipes (Expand to View Details)")
            for i, recipe in enumerate(retrieved_recipes):
                full_recipe = get_recipe_from_db(recipe["recipe_id"])
                if full_recipe:
                    with st.expander(f"{full_recipe.get('name', 'Unknown Recipe')}"):
                        # If the recipe has an image (URL in 'image'), display it
                        if full_recipe.get("image") and full_recipe["image"] != "N/A":
                            st.image(full_recipe["image"], 
                                     caption=full_recipe.get("name", "Recipe"))
                        
                        st.write("**Description:**", 
                                 full_recipe.get("description", ""))
                        st.write("**Ingredients:**", 
                                 full_recipe.get("ingredients", []))
                        st.write("**Steps:**", 
                                 full_recipe.get("steps", []))
                        st.write("**Total Time (mins):**", 
                                 full_recipe.get("total_time_mins", "N/A"))
                        st.write("**Difficulty:**", 
                                 full_recipe.get("difficulty", "N/A"))
                        st.write("**Ratings:**", 
                                 full_recipe.get("ratings", "N/A"))
                        st.write("**URL:**", 
                                 full_recipe.get("url", "N/A"))

    # Button to start a new chat
    if st.button("New Chat / Reset"):
        st.session_state["messages"] = []
        st.session_state["conversation_active"] = False

    # Chat interface if conversation is active
    if st.session_state["conversation_active"]:
        st.subheader("Chat with the Assistant")

        # Display existing conversation
        for i, msg in enumerate(st.session_state["messages"]):
            if msg["role"] == "assistant":
                st.markdown(f"**Assistant:** {msg['content']}")
            else:
                st.markdown(f"**You:** {msg['content']}")

        # User follow-up
        user_followup = st.text_input("Your message:")
        if st.button("Send"):
            if user_followup.strip():
                # Append user message
                st.session_state["messages"].append({"role": "user", "content": user_followup})
                
                # Prepare conversation for LLM
                conversation_for_llm = [
                    {"role": "system", "content": "You are a helpful recipe assistant."},
                ]
                for msg in st.session_state["messages"]:
                    conversation_for_llm.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

                # Call OpenAI again
                followup_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=conversation_for_llm,
                    temperature=0.4,
                    max_tokens=600
                )
                assistant_answer = followup_response.choices[0].message.content
                
                # Save assistant response
                st.session_state["messages"].append({"role": "assistant", "content": assistant_answer})
                
                st.experimental_rerun()

# -----------------------
# 4. RUN THE APP
# -----------------------
if __name__ == "__main__":
    main()
