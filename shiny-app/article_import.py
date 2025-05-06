from pathlib import Path

import pandas as pd
import json

app_dir = (Path(__file__).parent).parent
f = open(app_dir / 'data/article_data.json', encoding="utf8")
article_data = json.load(f)