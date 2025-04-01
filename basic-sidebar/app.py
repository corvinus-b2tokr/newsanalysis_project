# Import data from shared.py
from shared import df

from shiny.express import input, render, ui
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64

all_tags = set(tag for tags in df["tags"] if isinstance(tags, list) for tag in tags)

ui.page_opts(title="Word Cloud from JSON")

with ui.sidebar():
    ui.input_select("topic", "Choose Topic", choices=list(all_tags), selected=list(all_tags)[0]),
    ui.input_slider("max_words", "Maximum Words:", min=10, max=200, value=100)

def get_filtered_text(topic):
        filtered_texts = df[df["tags"].apply(lambda x: topic in x if isinstance(x, list) else False)]["article_text"].tolist()
        return " ".join(filtered_texts)
    
@render.ui
def wordcloud():
    topic = input.topic()
    text = get_filtered_text(topic)
    if not text.strip():
        return "No words to display."
        
    wordcloud = WordCloud(width=1000, height=500, max_words=input.max_words(), background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
        
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
        
    return ui.tags.img(src="data:image/png;base64," + img_base64, style="width:100%")
