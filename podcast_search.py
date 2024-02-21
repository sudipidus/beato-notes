from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict
from fastapi.responses import HTMLResponse
from fuzzywuzzy import fuzz
import psycopg2

app = FastAPI()

# Establish connection to PostgreSQL database
conn = psycopg2.connect(
    dbname="beato_notes",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5435"
)

# Define the fuzzy search function
def perform_fuzzy_search(query: str) -> List[Dict]:
    search_results = []
    cur = conn.cursor()
    cur.execute("SELECT id, url, timestamp_start, timestamp_end, text FROM podcasts")
    rows = cur.fetchall()
    for row in rows:
        text = row[4]  # Text is in the fifth column (index 4) in the database
        similarity = fuzz.partial_ratio(query, text)
        search_results.append((row, similarity))
    search_results.sort(key=lambda x: x[1], reverse=True)
    return [result[0] for result in search_results[:3]]  # Return only top 3 results

# Read HTML content from template file
with open("template.html", "r") as file:
    html_content = file.read()

# Define the API endpoint
@app.get("/search/")
async def search_text(text: str = Query(None, title="Search Text")):
    if not text:
        raise HTTPException(status_code=400, detail="Query parameter 'text' is required")
    results = perform_fuzzy_search(text)
    return results

# Define the UI endpoint
@app.get("/")
async def get_ui():
    return HTMLResponse(content=html_content, status_code=200, media_type="text/html")
