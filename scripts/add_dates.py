import json
from pathlib import Path

# Load the articles from the JSON file
input_file = Path("../data/article_data.json")
output_file = Path("../data/article_data_dates.json")

with open(input_file, encoding="utf8") as f:
    articles = json.load(f)

# Function to extract date from the URL
def extract_date_from_url(url):
    try:
        # Extract the date part (YYYY/MM/DD) from the URL
        parts = url.split("/")
        year, month, day = parts[-4], parts[-3], parts[-2]
        return f"{year}-{month}-{day}"
    except (IndexError, ValueError):
        return None  # Return None if the date cannot be extracted

# Add the date to each article
for article in articles:
    article["date"] = extract_date_from_url(article.get("url", ""))

# Save the updated articles to a new JSON file
with open(output_file, "w", encoding="utf8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"Updated articles saved to {output_file}")