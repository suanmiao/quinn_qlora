import re
from functools import wraps
from dotenv import load_dotenv
import os

# Specify the path to your .env file
dotenv_path = "/dbfs/quinn_leng/.env"

# Load the environment variables
load_dotenv(dotenv_path=dotenv_path)

# Now you can access the environment variables using os.getenv
openai_token = os.getenv('OPENAI_TOKEN')
openai_org = os.getenv('OPENAI_ORGANIZATION')
pat_token = os.getenv('PAT_TOKEN')

print(f"Got openai token: {openai_token}, openai org: {openai_org}, pat token: {pat_token}")

def validate_docstrings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Run the validation before executing the function
        has_errors = wrapper.validate_docstrings()
        if has_errors:
            print("Validation failed. Not executing the function.")
            return None

        return func(*args, **kwargs)

    def validate_docstrings_decorator():
        errors = []

        # Check if the function has a docstring
        docstring = func.__doc__
        if not docstring:
            errors.append(f"Method '{func.__name__}' is missing a docstring.")
        else:
            # Check if the function description is present
            if not docstring.strip().split("\n")[0].strip():
                errors.append(f"Method '{func.__name__}' is missing a description.")

            # Extract parameter descriptions
            param_pattern = re.compile(r'\s*(\w+)\s*\((\w+)\):\s*(.*)')
            param_descriptions = {match.group(1): match.group(3) for match in re.finditer(param_pattern, docstring)}

            # Check if all non-default parameters have a description
            for param in func.__annotations__:
                if param not in param_descriptions:
                    errors.append(f"Parameter '{param}' in method '{func.__name__}' is missing a description. Expected format: '{param} ({func.__annotations__[param]}): Description.'")

        for error in errors:
            print(f"Error: {error}")

        return len(errors) > 0

    wrapper.validate_docstrings = validate_docstrings_decorator
    return wrapper


import inspect
import json
import uuid
import threading
from abc import ABC, abstractmethod
from contextvars import ContextVar


class FunctionMeta(type):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if 'execute' in attrs:
            setattr(cls, 'execute', validate_docstrings(attrs['execute']))
        return cls

class AbstractFunction(metaclass=FunctionMeta):
    name = "abstract_function"
    description = "A abstract function."
    sources_dict = {}
    sources_lock = threading.Lock()
    uuid_context = ContextVar("uuid")  # Create a context variable for the UUID

    def append_sources(self, sources):
        uuid = self.uuid_context.get()  # Get the UUID from the context variable
        with self.sources_lock:
            if uuid not in self.sources_dict:
                self.sources_dict[uuid] = []
            self.sources_dict[uuid].extend(sources)

    def pop_sources(self):
        uuid = self.uuid_context.get()  # Get the UUID from the context variable
        with self.sources_lock:
            sources = self.sources_dict.pop(uuid, [])
        return sources

    def execute(self, *args, **kwargs):
        pass

    def export_as_openai_function(self):
        func_sig = inspect.signature(self.execute)
        parameters = {}
        required = []
        for name, param in func_sig.parameters.items():
            docstring = inspect.getdoc(self.execute)
            if docstring is not None and len(docstring.strip()) > 0:
              param_desc = next((line.split(":")[1].strip() for line in docstring.split("\n") if line.startswith(name + " :")), name)
              parameters[name] = {"type": "string", "description": param_desc}

            if param.default == inspect.Parameter.empty:
                required.append(name)
            
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            }
        }

    def dry_run(self):
        print("Running a dry run for the 'execute' method.")
        self.execute.validate_docstrings()

class FunctionAgent:
    def __init__(self, functions):
        self.function_map = {func.name: func for func in functions}
        self.uuid_context = ContextVar("uuid")  # Create a context variable for the UUID
        self.uuid_map = {}  # Create a map to store UUIDs

    def execute_function(self, json_blob):
        function_name = json_blob['name']
        arguments = json.loads(json_blob['arguments'])

        function = self.function_map[function_name]
        uuid_str = str(uuid.uuid4())  # Generate a unique UUID

        # Set the UUID in the function's context and store it in uuid_map
        token = function.uuid_context.set(uuid_str)
        self.uuid_map[function_name] = uuid_str
        try:
            result = function.execute(**arguments)
        finally:
            function.uuid_context.reset(token)

        return result, uuid_str

    def pop_sources(self):
        sources = []
        for function_name, function in self.function_map.items():
            # Check if the function has been executed and has a UUID
            if function_name in self.uuid_map:
                # Set the UUID in the context from the uuid_map before calling pop_sources
                token = function.uuid_context.set(self.uuid_map[function_name])
                try:
                    sources.extend(function.pop_sources())
                finally:
                    function.uuid_context.reset(token)
        return sources
        
    def export_as_openai_functions(self):
        return [function.export_as_openai_function() for function in self.function_map.values()]


import requests
import json
import time

def request_openai(messages, functions=[], temperature=0.3, model="gpt-4"):
  print(f"Calling open-ai API with {len(messages)} messages and {len(functions)} functions")

  token = openai_token
  url = "https://api.openai.com/v1/chat/completions"
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {token}",
      "OpenAI-Organization": openai_org,
  }
  data = {
      "model": model,
      "messages": messages,
      "temperature": temperature,
  }

  if len(functions) > 0:
    data["functions"] = functions
  response = requests.post(url, headers=headers, data=json.dumps(data))
  if response.status_code != 200:
    print(f"Got non-200 response from openAI, status code {response.status_code}, response: {response.text}")
    return None
  response_json = response.json()

  completion = response_json
  response_message = completion["choices"][0]["message"]
  return response_message


def request_openai_retry(messages, functions=[], temperature=0.3, model="gpt-4"):
  start_time = time.time()
  while time.time() - start_time < 1200:
    response = request_openai_proxy(messages=messages, functions=functions)
    if response != None:
      return response
    else:
      print(f"Error when requesting openAI, retrying after 2 seconds")
      time.sleep(2)
  return None

def request_openai_proxy(messages, functions=[], temperature=0.3, model="gpt-4"):
  print(f"Calling open-ai proxy with {len(messages)} messages and {len(functions)} functions")

  token = pat_token
  url = "https://e2-dogfood.staging.cloud.databricks.com/api/2.0/lakesense-v2/chat/completions"
  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
      "@method": "enterpriseOpenAiServiceChatCompletionRequest",
      "params": {
          "model": model,
          "messages": messages,
          "temperature": temperature,
      }
  }
  if len(functions) > 0:
    data["params"]["functions"] = functions
  response = requests.post(url, headers=headers, data=json.dumps(data))
  if response.status_code != 200:
    print(f"Got non-200 response from openAI, status code {response.status_code}, response: {response.text}")
  response_json = response.json()

  completion = json.loads(response_json['completion'])
  response_message = completion["choices"][0]["message"]
  return response_message
