
import pandas as pd
import json
from openai_utils import AbstractFunction, FunctionAgent, request_openai_retry

def eval_jsonl(file_path, concurrency=5, output_df_path=None):
    df = prediction_as_df(file_path)
    eval_result_df = run_eval(concurrency=concurrency, dataset_df=df)
    print(eval_result_df.head(5))
    eval_result_df["final_score"] = pd.to_numeric(eval_result_df["final_score"], errors='coerce')
    eval_result_df["correctness"] = pd.to_numeric(eval_result_df["correctness"], errors='coerce')
    eval_result_df["comprehensiveness"] = pd.to_numeric(eval_result_df["comprehensiveness"], errors='coerce')
    eval_result_df["readability"] = pd.to_numeric(eval_result_df["readability"], errors='coerce')


    final_score_mean = round(eval_result_df["final_score"].mean(), 2)
    correctness_mean = round(eval_result_df["correctness"].mean(), 2)
    comprehensiveness_mean = round(eval_result_df["comprehensiveness"].mean(), 2)
    readability_mean = round(eval_result_df["readability"].mean(), 2)

    print(f"Got final score {final_score_mean}, correctness {correctness_mean}, comprehensiveness {comprehensiveness_mean}, readability {readability_mean}")
    eval_result_df.to_csv(output_df_path, index=False)

def prediction_as_df(file_path):
    # Load data from the jsonl file
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    # Convert list of dictionaries to pandas dataframe
    df = pd.DataFrame(data)

    # Rename the columns
    df = df.rename(columns={"input": "question", "prediction": "provided_answer", "ref_output": "reference_answer"})

    return df


import pandas as pd

class GradingFunction(AbstractFunction):
    name = "grading_function"
    description = "Call this function to submit the grading for the answer"


    def execute(self, reasoning_for_correctness, score_for_correctness, reasoning_for_comprehensiveness, score_for_comprehensiveness, reasoning_for_readability, score_for_readability, final_score):
        """
        Call this function to submit the grading for the answer

        Parameters:
        reasoning_for_correctness : You reasoning for the giving the grading for the correctness of answer. Provide 5 to 30 words explaination. 
        score_for_correctness : You numerical (integer) grading between 1 to 5 for the correctness of the answer.
        reasoning_for_comprehensiveness : You reasoning for the giving the grading for the comprehensiveness of answer. Provide 5 to 30 words explaination. 
        score_for_comprehensiveness : You numerical (integer) grading between 1 to 5 for the comprehensiveness of the answer.
        reasoning_for_readability : You reasoning for the giving the grading for the readability of answer. Provide 5 to 30 words explaination. 
        score_for_readability : You numerical (integer) grading between 1 to 5 for the readability of the answer.
        final_score : the final rating, which is 60% correctness + 20% comprehensiveness + 20% readability. In float format. 
        Returns: it will always return ok. 
        """      


        eval_result = {
          "final_score": final_score,
          "correctness": score_for_correctness,
          "reasoning_for_correctness": reasoning_for_correctness,
          "comprehensiveness": score_for_comprehensiveness,
          "reasoning_for_comprehensiveness": reasoning_for_comprehensiveness,
          "readability": score_for_readability,
          "reasoning_for_readability": reasoning_for_readability
        }
        self.append_sources([eval_result])

        return "ok"
      

# Initialize function objects
grading_func = GradingFunction()
grading_func.dry_run()


# Initialize function agent with the function objects
grading_agent = FunctionAgent([grading_func])
print(grading_agent.export_as_openai_functions() )

def grading_answer(question, reference_answer, provided_answer):
  import requests
  import json
  
  functions = grading_agent.export_as_openai_functions()  

  system_prompt = """ Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on the question and the reference answer.

  You'll be given a function grading_function which you'll call for each provided question, reference answer and provided answer to submit your reasoning and score for the correctness, comprehensiveness and readability of the answer. 
  
  Below is your grading rubric: 

  - Correctness
    - If the answer correctly answer the question, and you can use the provided reference answer for checking correctness.
- Comprehensiveness:
    - How comprehensive is the answer, does it fully answer all aspects of the question and provide comprehensive explanation and other necessary information.
- Readability: (avoid repeat)
    - How readable is the answer, does it have redundant information or incomplete information that hurts the readability of the answer.
- Then final rating:
    - Ratio: 60% correctness + 20% comprehensiveness + 20% readability
  """

  messages = [
    {"role": "system", "content": system_prompt},
    ]

  grading_material = f"""
Provided question:
{question}

Reference answer: 
{reference_answer}

Provided answer: 
{provided_answer}

"""
  messages.append({"role": "user", "content": grading_material})

  response_message = request_openai_retry(messages, functions)
  content = response_message["content"]
  print(f"Received content {content}")
  if "function_call" in response_message:
    function_call_obj = response_message["function_call"]
    function_name = function_call_obj["name"]
    print(f"Calling function {function_call_obj}")
    grading_agent.execute_function(function_call_obj)
    eval_result = grading_agent.pop_sources()[0]
    # print(f"Got eval result {eval_result}")
    eval_result["question"] = question
    eval_result["provided_answer"] = provided_answer
    eval_result["reference_answer"] = reference_answer
    return eval_result
  else:
    return None
  
import concurrent.futures

def grade_row(row):
    try:
        question = row['question']
        reference_answer = row['reference_answer']
        provided_answer = row['provided_answer']
        eval_result = grading_answer(question=question, provided_answer=provided_answer, reference_answer=reference_answer)
        status = "ok"
        failure_msg = ""
    except Exception as e:
        eval_result = {
            "final_score": None, 
            "correctness": None, 
            "reasoning_for_correctness": None, 
            "comprehensiveness": None, 
            "reasoning_for_comprehensiveness": None, 
            "readability": None, 
            "reasoning_for_readability": None
        }
        status = "fail"
        failure_msg = str(e)
    
    return {
      "reference_answer": reference_answer, 
      "question": question, 
      "provided_answer": provided_answer, 
      "final_score": eval_result["final_score"], 
      "correctness": eval_result["correctness"], 
      "reasoning_for_correctness": eval_result["reasoning_for_correctness"], 
      "comprehensiveness": eval_result["comprehensiveness"], 
      "reasoning_for_comprehensiveness": eval_result["reasoning_for_comprehensiveness"], 
      "readability": eval_result["readability"], 
      "reasoning_for_readability": eval_result["reasoning_for_readability"], 
      "status": status, 
      "failure_msg": failure_msg
    }


def run_eval(concurrency=3, dataset_df=None):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_row = {executor.submit(grade_row, row): row for index, row in dataset_df.iterrows()}
        for future in concurrent.futures.as_completed(future_to_row):
            row = future_to_row[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f'An exception was raised, {exc}')

    eval_result_df = pd.DataFrame(rows)
    return eval_result_df