from dotenv import load_dotenv
import tiktoken
from functools import lru_cache
import os

load_dotenv(os.path.join(os.path.expanduser('~'), '.gpt_ncats.env'))

LLM_MODEL = 'gpt-3.5-turbo'
TEMPERATURE = 0.7
MAX_TOKEN_LIMIT = 15000

HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT_WITH_JSON_RESPONSE = '''
You are an expert in summarizing biomedical knowledge in max 500 words.

You will receive a dictionary in the following format:
{
  "query": "<a question of string datatype>",
  "results": {
    "<name of a biomedical concept>": [
      <a list where each element represents a biological path (i.e., a sequence of edges) extracted from a knowledge graph>
    ]
  }
}

In the results dictionary:
<name of a biomedical concept> forms an answer to the query.
The corresponding list of biological paths represents the mechanistic explanation of how that answer (i.e., the biomedical concept) relates to the query.

Your task is to provide a summarized answer to the given query in 500 words. Do NOT list the results, instead you are supposed to give a biologically sound summarization for them in one paragraph. For that:

- Combine all the biomedical paths corresponding to all answers into a coherent textual summary that explains the mechanistic rationale for the given answer to the query.
- Summarize any shared paths or nodes by integrating them where applicable.
- Ensure that the summary is biologically accurate, by providing right explanations to the biomedical concepts such as genes or proteins.
- If no context is provided, report that there is no context available for summarization.

Your response should be in JSON format as follows:
{
  "query": "<given query>",
  "summary": "<the summary by combining all the given answers and their associated context>"
}

Example:
{
    "query": "what drugs may treat conditions related to Ehlers-Danlos syndrome, hypermobility type?"
    "cyclosporine": ['cyclosporine-[treats or applied or studied to treat]-Arthralgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type', 
                    'cyclosporine-[treats or applied or studied to treat]-xerophthalmia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type', 
                    'cyclosporine-[causes increased activity of]-COL3A1-[gene associated with condition]-Ehlers-Danlos syndrome, hypermobility type', 
                    'cyclosporine-[treats or applied or studied to treat]-gingival overgrowth-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type'],
    "cortisol": ['cortisol-[in clinical trials for]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[preventative for condition]-Fatigue-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[in clinical trials for]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type', 
                 'cortisol-[treats or applied or studied to treat]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-Fatigue-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[affects]-TGFB1-[physically interacts with]-COL5A1-[gene associated with condition]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type'],
    "lidocaine": ['lidocaine-[preventative for condition]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-scoliosis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-severe cutaneous adverse reaction-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']
}
Response:
{
  "query": "what drugs may treat conditions related to Ehlers-Danlos syndrome, hypermobility type?",
  "summary": "Cyclosporine is being studied for its potential to treat arthralgia, xerophthalmia, and gingival overgrowth, all of which are phenotypes associated with Ehlers-Danlos syndrome, hypermobility type. It also causes an increase in the activity of the COL3A1 gene related to this condition. Cortisol, on the other hand, is being evaluated in clinical trials for osteoarthritis and is considered as a preventative or treatment for fatigue, depressive disorder, anxiety, migraine disorder, and chronic pain in individuals with Ehlers-Danlos syndrome, hypermobility type. It also interacts with the TGFB1 gene, which physically interacts with COL5A1, another gene associated with this syndrome. Lidocaine shows promise in preventing or treating myalgia, depressive disorder, anxiety, osteoarthritis, cardiac rhythm disease, migraine disorder, scoliosis, chronic pain, severe cutaneous adverse reactions, and myalgia in individuals with Ehlers-Danlos syndrome, hypermobility type."
}
'''

HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT_WITHOUT_JSON_RESPONSE = '''
You are an expert in summarizing biomedical knowledge in max 500 words.

You will receive a dictionary in the following format:
{
  "query": "<a question of string datatype>",
  "results": {
    "<name of a biomedical concept>": [
      <a list where each element represents a biological path (i.e., a sequence of edges) extracted from a knowledge graph>
    ]
  }
}

In the results dictionary:
<name of a biomedical concept> forms an answer to the query.
The corresponding list of biological paths represents the mechanistic explanation of how that answer (i.e., the biomedical concept) relates to the query.

Your task is to provide a summarized answer to the given query in 500 words. Do NOT list the results, instead you are supposed to give a biologically sound summarization for them in one paragraph. For that:

- Combine all the biomedical paths corresponding to all answers into a coherent textual summary that explains the mechanistic rationale for the given answer to the query.
- Summarize any shared paths or nodes by integrating them where applicable.
- Ensure that the summary is biologically accurate, by providing right explanations to the biomedical concepts such as genes or proteins.
- If no context is provided, report that there is no context available for summarization.


Example:
{
    "query": "what drugs may treat conditions related to Ehlers-Danlos syndrome, hypermobility type?"
    "cyclosporine": ['cyclosporine-[treats or applied or studied to treat]-Arthralgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type', 
                    'cyclosporine-[treats or applied or studied to treat]-xerophthalmia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type', 
                    'cyclosporine-[causes increased activity of]-COL3A1-[gene associated with condition]-Ehlers-Danlos syndrome, hypermobility type', 
                    'cyclosporine-[treats or applied or studied to treat]-gingival overgrowth-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type'],
    "cortisol": ['cortisol-[in clinical trials for]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[preventative for condition]-Fatigue-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[in clinical trials for]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type', 
                 'cortisol-[treats or applied or studied to treat]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-Fatigue-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[affects]-TGFB1-[physically interacts with]-COL5A1-[gene associated with condition]-Ehlers-Danlos syndrome, hypermobility type',
                 'cortisol-[treats or applied or studied to treat]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type'],
    "lidocaine": ['lidocaine-[preventative for condition]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-scoliosis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-severe cutaneous adverse reaction-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']
}
Response:
Cyclosporine is being studied for its potential to treat arthralgia, xerophthalmia, and gingival overgrowth, all of which are phenotypes associated with Ehlers-Danlos syndrome, hypermobility type. It also causes an increase in the activity of the COL3A1 gene related to this condition. Cortisol, on the other hand, is being evaluated in clinical trials for osteoarthritis and is considered as a preventative or treatment for fatigue, depressive disorder, anxiety, migraine disorder, and chronic pain in individuals with Ehlers-Danlos syndrome, hypermobility type. It also interacts with the TGFB1 gene, which physically interacts with COL5A1, another gene associated with this syndrome. Lidocaine shows promise in preventing or treating myalgia, depressive disorder, anxiety, osteoarthritis, cardiac rhythm disease, migraine disorder, scoliosis, chronic pain, severe cutaneous adverse reactions, and myalgia in individuals with Ehlers-Danlos syndrome, hypermobility type.
'''



@lru_cache(maxsize=None)
def count_tokens_cached(text, model):
    return count_tokens(text, model)

def get_token_limited_prompt(query, system_prompt):
    results = list(query["results"].items())
    tot_results = len(results)
    
    def create_prompt(num_results):
        results_dict = dict(results[:num_results])
        return f'''
        {{
        "query": {query["query"]},
        "results": {results_dict}
        }}
        '''

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
    else:
        raise ValueError(f'{model} model not supported for encoding in count_tokens function')

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        print(f'Unable to count tokens from input \"{text}\". Error: {e}')
        return 0
