from flask import Flask, Response, jsonify, request
from webargs.flaskparser import use_kwargs
from webargs.fields import Str
from webargs import fields, validate
from openai import OpenAI
import json
from utility import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    endpoints = sorted(r.rule for r in app.url_map.iter_rules())
    return jsonify(
        endpoints=endpoints
    )

@app.route("/summary", methods=["POST"])
def summary():
    data = request.json
    if not isinstance(data, dict) or "query" not in data or "results" not in data:
        return jsonify(error="Invalid request format. Must contain 'query' and 'results' keys."), 400

    prompt = get_token_limited_prompt(data)
    
    def generate():
        stream = client.chat.completions.create(        
            temperature=TEMPERATURE,
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)