
from __future__ import print_function


import models.category_model
import models.emoji.emoji_model
import models3.safe_model
import models.related.related_model

import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

emoji_graph = models.emoji.emoji_model.EmojiModel()

@app.route('/analyze-emoji', methods=['POST'])
def analyze_emoji():
    input_data=request.json
    if len(input_data)<2:
      return '{"error":"at least 2 sentences required as input"}'

    result=emoji_graph.eval(input_data)
    return json.dumps(result)
    #return 'Hello, World!'

