## Biomedical Translator Data Summarization API

This repository contains a Flask-based API that generates summaries of structured responses from the [Biomedical Data Translator system](https://ui.transltr.io/demo) in natural language.

Currently, the API is hosted in the following URL:

https://biosummary.pythonanywhere.com

### Summary Endpoint

API has the summary endpoint called **"/summary"**.

It currently accepts JSON structured data which holds the user "query" and the "results" which has the following structure:

```
{
  "query": "<a question of string datatype>",
  "results": {
    "<name of a biomedical concept>": [
      <a list where each element represents a biological path (i.e., a sequence of edges) extracted from a knowledge graph>
    ]
  }
}
```

The list of biological paths represents the mechanistic explanation of how that answer (i.e., the biomedical concept) relates to the query.

#### Make an API call

**/summary** endpoint accepts **POST** request. Here is a snippet for calling the endpoint from python:

```
import requests

api_endpoint = "https://biosummary.pythonanywhere.com/summary"

def stream_response(url, data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers, stream=True)
    
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                print(chunk, end='', flush=True)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

stream_response(api_endpoint, data)
```

**Note:** 'data' is the JSON data as mentioned above

Refer [demo](https://github.com/karthiksoman/ncats_llm_summarization/blob/main/demo.ipynb) notebook for more details.
