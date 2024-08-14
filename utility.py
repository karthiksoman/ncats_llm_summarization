from dotenv import load_dotenv
import tiktoken
from functools import lru_cache
import os
import json

load_dotenv(os.path.join(os.path.expanduser('~'), '.gpt_ncats.env'))

LLM_MODEL = 'gpt-4o-mini'
TEMPERATURE = 0.7
MAX_TOKEN_LIMIT = 15500


HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT_WITHOUT_JSON_RESPONSE = '''
You are an expert in summarizing biomedical knowledge in max 1000 words.

You will receive a dictionary in the following format:
{
  "query": "<a question of string datatype>",
  "results": <a list of dictionary>
}

As mentioned in the above JSON format, "results" key will have a list of dictionary, where each dictionary will be of the following format:
{
  "<name of a biomedical concept>": [
      <a list where each element represents a biological path (i.e., a sequence of edges) extracted from a knowledge graph>
  ]
}
Hence, combining everything, the input JSON schema looks as follows:
{
  "query": "<a question of string datatype>",
  "results": [
    {
      "<name of a biomedical concept>": [
        "<biological path 1>",
        "<biological path 2>",
        "<biological path 3>"
      ]
    }
  ]
}

<name of a biomedical concept> forms an answer to the "query".
The corresponding list of biological paths represents the mechanistic explanation of how that answer (i.e., the biomedical concept) relates to the query.

Your task is to provide a summarized answer to the given query in max 1000 words. Do NOT list the results, instead you are supposed to give a biologically sound summarization for them in one paragraph. For that:

- Combine all the biological paths corresponding to all answers into a graph and then provide a coherent textual summary that explains the mechanistic rationale for the given answers to the query.
- Summarize any shared paths or nodes by integrating them where applicable.
- Ensure that the summary is biologically accurate, by providing right explanations to the biomedical concepts such as genes or proteins.
- If no context is provided, report that there is no context available for summarization.


Example:
{
    "query": "what drugs may treat conditions related to Ehlers-Danlos syndrome, hypermobility type?"
    "results" : [
                {'name': 'cyclosporine',
                 'paths': ['cyclosporine-[causes increased activity of]-COL3A1-[gene associated with condition]-Ehlers-Danlos syndrome, hypermobility type',
                  'cyclosporine-[treats or applied or studied to treat]-gingival overgrowth-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'cyclosporine-[treats or applied or studied to treat]-Arthralgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'cyclosporine-[treats or applied or studied to treat]-xerophthalmia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']},
                {'name': 'citalopram',
                 'paths': ['citalopram-[in clinical trials for]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[treats]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[applied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[applied to treat]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[in clinical trials for]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']},
                {'name': 'lidocaine',
                 'paths': ['lidocaine-[preventative for condition]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-scoliosis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-severe cutaneous adverse reaction-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']}
          ]
}

Response:
Ehlers-Danlos syndrome, hypermobility type (hEDS) is characterized by joint hypermobility, skin extensibility, and a range of related complications, including chronic pain, anxiety, and dysautonomia. Various pharmacological treatments have been investigated for addressing the symptoms and associated conditions of hEDS. Cyclosporine has been identified as a potential treatment as it may lead to increased activity of the COL3A1 gene, which is relevant in collagen formation and may reduce specific symptoms such as gingival overgrowth and arthralgia alongside xerophthalmia. Citalopram, a selective serotonin reuptake inhibitor (SSRI), is being studied primarily for treating anxiety and depressive disorders, which are common in individuals with hEDS, potentially enhancing overall emotional well-being and management of pain. Lidocaine serves multiple roles, primarily in pain management; it has been investigated for its efficacy in treating chronic pain, myalgia, cardiac rhythm disturbances, and even migraine disorders within the context of hEDS. Clinical trials are ongoing to evaluate its effectiveness for anxiety and other associated conditions, highlighting its versatility in managing multiple phenotypes of this syndrome. These pharmacological interventions aim to alleviate not only the physical manifestations of hEDS but also the psychological distress associated with enduring chronic symptoms, emphasizing a multi-faceted approach to managing this complex disorder.
'''



@lru_cache(maxsize=None)
def count_tokens_cached(text, model):
    return count_tokens(text, model)

def get_token_limited_prompt(query, system_prompt):
    results = query["results"]
    tot_results = len(results)
    
    def create_prompt(num_results):
        limited_results = results[:num_results]
        prompt_data = {
            "query": query["query"],
            "results": limited_results
        }
        return json.dumps(prompt_data, indent=2)

    def is_within_limit(num_results):
        prompt = create_prompt(num_results)
        full_prompt = system_prompt + '\n' + prompt
        return count_tokens_cached(full_prompt, LLM_MODEL) <= MAX_TOKEN_LIMIT

    # Binary search
    left, right = 1, tot_results
    while left <= right:
        mid = (left + right) // 2
        if is_within_limit(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return create_prompt(right)

def count_tokens(text: str, model: str) -> int:    
    if 'gpt-4' in model.lower() or 'gpt-35-turbo' in model.lower() or 'gpt-3.5-turbo' in model.lower() or model == 'text-embedding-ada-002':
        encoding_name = 'cl100k_base'
    elif model == 'text-davinci-003':
        encoding_name = 'p50k_base'
    elif 'gpt-4o' in model:
        encoding_name = 'o200k_base'
    else:
        raise ValueError(f'{model} model not supported for encoding in count_tokens function')
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        print(f'Unable to count tokens from input \"{text}\". Error: {e}')
        return 0