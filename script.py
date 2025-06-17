REFERENCE_YEAR = 2025
REFERENCE_MONTH = 8

# #### Lees het maandelijkse bestand in

import pandas as pd
import os
from azure.storage.blob import BlobClient
import os


api_key = os.environ['OPENAI_KEY']

# Function to read a sheet and trim based on first column's content
def read_clean_sheet(file_path, sheet_name):
    # Read full sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Skip the second row by dropping it (assuming row 0 = header, row 1 = skip)
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[1:,:]

    # Determine the index of the first invalid row (NaN, empty string, or space)
    first_col = df.iloc[1:, 0]
    invalid_mask = first_col.isna() | (first_col.astype(str).str.strip() == "")
    if invalid_mask.any():
        end_index = invalid_mask.idxmax()  # index of the first True
        df = df.loc[:end_index - 1]  # exclude the invalid row
    else:
        # If no invalid row found, keep the full DataFrame
        pass
    
    #df["filiaal"] = sheet_name

    return df

# List of sheet names
sheet_names = ["west 1", "west 2", "west3bg", "zuid", "oost 1", "oost 2", "oost3bg", "aanleun"]

#Users/jordi.vanselm/Roosterdata HHH_8.xlsx

# File path
file_path = f"Roosterdata HHH_{REFERENCE_MONTH}.xlsx"
#file_path = "Roosterdata_HHH_ingeladen.xlsx"

# Load all sheets dynamically
dfs = {name: read_clean_sheet(file_path, name) for name in sheet_names}

emp_file = pd.concat(
    [df.assign(filiaal=name) for name, df in dfs.items()],
    ignore_index=True
)

emp_file = pd.DataFrame(emp_file)
emp_file['filiaal'] = emp_file['filiaal'].str.replace(' ', '', regex=False).str.lower() # converteer all sheetnamen naar gestandaardiseerde afdelingnamen. 

emp_num  = emp_file.shape[0]
emp_num_split = len(emp_file['harde wens'].str.cat(sep=';', na_rep='NaN').split(';'))

print(emp_file.columns)
print(emp_file[emp_file["naam"] == "Aafke de Boer"]["contract uren"])


# ## Verwerk tekstuele kolommen met AI

# #### Initieer de OpenAI client en output schema's
# Call ChatGPT API
from openai import OpenAI
from pydantic import BaseModel, field_validator, RootModel, conlist
from typing import List, Union, Dict
import time
import json
import os

max_retries = 20
attempt = 0
timeout = 20

client = OpenAI(
    api_key=api_key,
)
class WensenVrij(BaseModel):
  vrije_dagen: list[str] 
    
  @field_validator("vrije_dagen")
  @classmethod
  def check_vrije_dagen(cls, v):
      if len(v) != emp_num_split:  # Bijvoorbeeld, minimaal 3 vrije dagen vereist
          raise ValueError("Onvoldoende vrije dagen teruggegeven.")
      return v
        
class VakantiedagenMaand(BaseModel):
    vakantiedagen: List[List[Union[int, str]]]  # Allow integers for days, and "NaN" as a string
    
    class Config:
        from_attributes = True  # Allows compatibility with ORM objects
        
class VrijeDagenNVMaand(BaseModel):
    vrije_dagen_nv: List[List[Union[int, str]]]  # Allow integers for days, and "NaN" as a string
    
    class Config:
        from_attributes = True  # Allows compatibility with ORM objects
        

class WensenVrij(BaseModel):
    vrije_dagen: List[List[Union[int, str]]]  # Each employee gets a list of free weekdays (1-7) or "NaN"

    class Config:
        from_attributes = True  # ✅ Pydantic v2 fix




class Dayblock(BaseModel):
    Dayblock : Dict[str, List[int]]

class EmployeeSchedule(BaseModel):
    EmployeeSchedule: List[Dayblock]

# Create custom schema for day and time nested lists with dictionaries for each day
def create_day_time_tool(schema_name) : 
    tool = {
            "type": "json_schema",

            "json_schema": 
                {

                "name": "Nested_dag_tijd_schema",

                "strict": True,

                "schema": {

                    "type": "object",

                    'properties': 
                        {schema_name: 
                            {'items': 
                                {'items': 
                                    {'type': 'string'}
                                , 'type': 'string'}, 
                            'title': schema_name,
                            'type': 'array'
                            }
                        }, 
                        
                    'required': [schema_name],
                    'additionalProperties': False,
                        }


                    }
                }

    return tool

tool_overleg = create_day_time_tool("Overleg")
tool_scholing = create_day_time_tool("Scholing")
tool_tijd    = create_day_time_tool("TijdWensen")


# Function to improve robustness of outputs:

import pandas as pd
import math

def get_optimal_batch_size(n_rows, max_batch_size=10, min_tail_size=3):
    """
    Determine the best batch size such that:
    - Each batch has at most max_batch_size rows
    - The final batch has at least min_tail_size rows
    """
    for batch_size in range(max_batch_size, 0, -1):
        num_full_batches = n_rows // batch_size
        remainder = n_rows % batch_size
        if remainder == 0 or remainder >= min_tail_size:
            return batch_size
    raise ValueError("Could not find a valid batch size satisfying constraints.")

n = len(emp_file)
batch_size = get_optimal_batch_size(n, max_batch_size=10, min_tail_size=3)

def prepare_prompt_for_translation(column: pd.Series, description: str = "values to translate", batch_size =  batch_size) -> tuple[str, dict]:
    """
    Prepares a prompt for GPT translation by extracting non-NaN values from a pandas Series,
    including their original indices and count, and formats them into a prompt string.

    Parameters:
        column (pd.Series): The input Series with possible NaN values.
        description (str): Optional description of what is being translated (e.g. "days").

    Returns:
        tuple:
            - str: Prompt string ready to send to GPT.
            - dict: Mapping of original indices to their corresponding values.
    """
    non_nan_series = column.dropna()
    index_value_map = {k % batch_size: v for k, v in non_nan_series.to_dict().items()}
    count = len(index_value_map)

    prompt = (
        f"There are {count} {description}. "
        f"Here are the {description}, as a Python dictionary where the keys are the indices from this batch of data that had non-empty values:\n\n"
        f"{index_value_map}\n\n"
        f"Take great in care in ensuring that the output list contains at most {batch_size} values and the indeces presented here are also the indices of the output lists that you populate."
      )

    return prompt, index_value_map

print(prepare_prompt_for_translation(emp_file["harde wens"].iloc[10:20]))

def parse_and_validate_tijdwensen(batch_output, batch_size, non_empty_input_dict):
    if len(batch_output) != batch_size:
        raise ValueError("Output length mismatch")
    
    non_empty_indices = set(non_empty_input_dict.keys())


    validated = []
    for idx, item in enumerate(batch_output):
        original_item = item  # Save for error reporting


        # Step 1: If empty string, treat as empty list
        if item == "" or item == " ":
            item = []

        # Step 2: If string, attempt to parse
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception as e:
                raise ValueError(f"Index {idx} not parseable: {e}. Content: {repr(original_item)}")

        # Step 3: Now check type
        if isinstance(item, list):
            if not all(isinstance(d, dict) for d in item):
                raise ValueError(f"Not all elements in item {idx} are dictionaries. Content: {repr(item)}")
            
            # For non-empty input, output must not be empty
            if idx in non_empty_indices and len(item) == 0:
                input_excerpt = non_empty_input_dict[idx]
                raise ValueError(
                    f"Row {idx} had input '{input_excerpt}' but produced empty output {batch_output}."
                )

            validated.append(item)
        else:
            raise ValueError(f"Index {idx} invalid format after parsing. Got {type(item)} instead of list. Content: {repr(item)}")

    return validated


# #### Gebruik AI om de kolommen met vrije dagen om te zetten naar code-suited inputs

# In[13]:


all_wensen_vrij = []  # Store results

for batch_start in range(0, len(emp_file), batch_size):
    batch = emp_file.iloc[batch_start : batch_start + batch_size]
    print(batch["harde wens"])
    
    # Initialise a prompt specifying the amount of values that were non-empty inside of the batch to improve robustness
    prepared_prompt, _ = prepare_prompt_for_translation(batch["harde wens"], batch_size=len(batch))
    #print(prepared_prompt)
    
    attempt = 0
    while attempt < max_retries:
        try:
            completion_WensenVrij = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                store=True,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": f"""
                        {prepared_prompt} 
                        Each entry is a field from a data frame column called 'harde wens' describing preferred free weekdays in natural language in Dutch. 
                        Your task is to return a JSON with exactly {len(batch)} entries in 'vrije_dagen', where:
                        - Each item in 'vrije_dagen' is a list of numbers (1-7) representing weekdays.
                        - Monday = 1, Sunday = 7.
                        - The output MUST contain exactly {len(batch)} lists, one per employee.
                        - If an employee's input contains 'NaN', return [] for that entry.
                        - If no free days are mentioned, return an empty list [] for that employee.
                        - Example output:
                          ```json
                          {{
                            "vrije_dagen": [
                              [1, 3, 5],  # Employee 1 (Monday, Wednesday, Friday)
                              [],           # Employee 2 (No preference)
                              [4],        # Employee 3 (Thursday)
                              []          # Employee 4 (No preference)
                            ]
                          }}
                          ```
                          IMPORTANT: the natural language describing the free week days can be denoted in multiple fashsions. For instance, only the weekday name(s) can be mentioned (e.g. Maandag, zaterdag), but also phrases like 'Maandag en zaterdag vrij" can occur or some variation of this.
                          In these cases, the same wish is implied. In this example, the result [1,6] should be returned. 
                    """},
                    {"role": "user", "content": json.dumps(batch[['harde wens']].to_dict(orient='records'))}
                ],
                response_format=WensenVrij,  # ✅ Uses updated schema
            )

            # Extract JSON safely
            response_json = json.loads(completion_WensenVrij.choices[0].message.content)
            print(response_json.get("vrije_dagen"))
            
            # Ensure output length matches input length
            if len(response_json.get("vrije_dagen")) != len(batch):
                raise ValueError("Mismatch in output length!")

            all_wensen_vrij.extend(response_json["vrije_dagen"])  # Store results
            break  # Success, move to next batch

        except Exception as e:
            attempt += 1
            print(f"Error occurred in batch {batch_start}-{batch_start+batch_size}: {e}. Attempt {attempt}/{max_retries}")
            time.sleep(2)

# If max retries reached, raise error
if attempt == max_retries:
    raise RuntimeError("Max retry attempts reached. Exiting.")


# #### Gebruik AI om de kolommen met vakantiedagen om te zetten naar code-suited inputs

# In[14]:


### Retrieve the holiday periods for the employees
## Retrieve the off-days for the employees

attempt = 0
all_vacation_days = []  # Store results
max_retries = 50


for batch_start in range(0, len(emp_file), batch_size):
    batch = emp_file.iloc[batch_start : batch_start + batch_size,]
    print(batch[["vakantie"]])

    # Initialise a prompt specifying the amount of values that were non-empty inside of the batch to improve robustness
    prepared_prompt, _ = prepare_prompt_for_translation(batch["vakantie"], batch_size=len(batch))
    print(prepared_prompt)
    attempt = 0
    while attempt < max_retries:
        try:
            completion_vakantie = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                store=True,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": f"""
                        {prepared_prompt} 
                        Each entry is a field from a data frame column called 'vakantie' describing vacation days in natural language. 
                        Your task is to return a JSON with exactly {len(batch)} entries in 'vakantiedagen', where:
                        - Each item in 'vakantiedagen' is a list of individual vacation days (as integers from 1-31).
                        - The output MUST contain exactly {len(batch)} lists, one per employee.
                        - If an employee's input contains 'NaN', return [] for that entry.
                        - If no vacation is mentioned, return an empty list [] for that employee.
                        - Example output (example month is juni / june):  
                        ```json
                        {{
                            "vakantiedagen": [
                            [7, 10, 11, 12],  # Employee 1 input: '7 en 10 tm 12 juni'  (or some other month)
                            [1, 2, 3],  # Employee 2 input: '1 tm 3 juni' 
                            [15, 16, 17],  # Employee 3 input: 15 tm 17 juni
                            [4, 5, 6, 7, 19, 24] # Employee 4 input: '4 tm 7 juni, 19 en 24 juni
                            []  # Employee 4 (no vacation)
                            ]
                        }}
                        ```
                        
                        IMPORTANT: When a input is given like '1 tm 5', or '20 tm 29', make sure to include the final date as well. IT IS NOT A PYTHON INDEX (hence, the examples would include values 5 and 29 respectively)!
                    """},
                    {"role": "user", "content": json.dumps(batch[['vakantie']].to_dict(orient='records'))}
                ],
                response_format=VakantiedagenMaand,
            )

            # Extract JSON safely
            response_json = json.loads(completion_vakantie.choices[0].message.content)
            print(response_json["vakantiedagen"])
            # Ensure output length matches input length
            if len(response_json["vakantiedagen"]) != len(batch):
                raise ValueError(f"Mismatch in output length! {len(response_json['vakantiedagen'])} vs {len(batch)}")

            all_vacation_days.extend(response_json["vakantiedagen"])  # Store results
            break  # Success, move to next batch

        except Exception as e:
            attempt += 1
            print(f"Error occurred in batch {batch_start}-{batch_start+batch_size}: {e}. Attempt {attempt}/{max_retries}")
            time.sleep(2)

# If max retries reached, raise error
if attempt == max_retries:
    raise RuntimeError("Max retry attempts reached. Exiting.")


# ### Vrije dagen (niet verlof)

# In[15]:


### Retrieve the holiday periods for the employees
## Retrieve the off-days for the employees that are not paid leave

attempt = 0

all_vrije_nv_days = []  # Store results

for batch_start in range(0, len(emp_file), batch_size):
    batch = emp_file.iloc[batch_start : batch_start + batch_size,]
    print(batch[["vrije dagen (niet verlof)"]])

    # Initialise a prompt specifying the amount of values that were non-empty inside of the batch to improve robustness
    prepared_prompt, _ = prepare_prompt_for_translation(batch["vrije dagen (niet verlof)"], batch_size=len(batch))
    
    attempt = 0
    while attempt < max_retries:
        try:
            completion_vrije_dagen_nv = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                store=True,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": f"""
                        {prepared_prompt} 
                        Each entry is a field from a data frame column called 'vrije dagen (niet verlof)' describing vacation days in natural language. 
                        Your task is to return a JSON with exactly {len(batch)} entries in 'vrije_dagen_nv', where:
                        - Each item in 'vrije_dagen_nv' is a list of individual vacation days (as integers from 1-31).
                        - The output MUST contain exactly {len(batch)} lists, one per employee.
                        - If an employee's input contains 'NaN', return [] for that entry.
                        - If no vacation is mentioned, return an empty list [] for that employee.
                        - Example output (example month is juni / june):  
                        ```json
                        {{
                            "vrije_dagen_nv": [
                            [7, 10, 11, 12],  # Employee 1 input: '7 en 10 tm 12 juni'  (or some other month)
                            [1, 2, 3],  # Employee 2 input: '1 tm 3 juni' 
                            [15, 16, 17],  # Employee 3 input: 15 tm 17 juni
                            [4, 5, 6, 7, 19, 24] # Employee 4 input: '4 tm 7 juni, 19 en 24 juni
                            []  # Employee 4 (no vacation)
                            ]
                        }}
                        ```
                        
                        IMPORTANT: When a input is given like '1 tm 5', or '20 tm 29', make sure to include the final date as well. IT IS NOT A PYTHON INDEX (hence, the examples would include values 5 and 29 respectively)!
                    """},
                    {"role": "user", "content": json.dumps(batch[["vrije dagen (niet verlof)"]].to_dict(orient='records'))}
                ],
                response_format=VrijeDagenNVMaand,
            )

            # Extract JSON safely
            response_json = json.loads(completion_vrije_dagen_nv.choices[0].message.content)
            print(response_json)
            print(response_json["vrije_dagen_nv"])
            # Ensure output length matches input length
            if len(response_json["vrije_dagen_nv"]) != len(batch):
                raise ValueError(f"Mismatch in output length! {len(response_json['vrije_dagen_nv'])} vs {len(batch)}")

            all_vrije_nv_days.extend(response_json["vrije_dagen_nv"])  # Store results
            break  # Success, move to next batch

        except Exception as e:
            attempt += 1
            print(f"Error occurred in batch {batch_start}-{batch_start+batch_size}: {e}. Attempt {attempt}/{max_retries}")
            time.sleep(2)

# If max retries reached, raise error
if attempt == max_retries:
    raise RuntimeError("Max retry attempts reached. Exiting.")


# #### Gebruik AI om de kolommen met scholingstijden om te zetten naar code-suited inputs

# In[16]:


### Retrieve the scholingstijden for the employees
### De default logica zorgt ervoor dat bij 8 urige schooldagen de alle shifts worden geblokkeerd doordat er geen shifts zijn die op die dag kunnen beginnen of eindigen tussen 6 and 17 uur. 
attempt = 0

overleg_tijden = []  # Store results

for batch_start in range(0, len(emp_file), batch_size):
    batch = emp_file.iloc[batch_start : batch_start + batch_size,]
    print(batch[["overleg"]].fillna("[]")) # add a dash to prevent truly empty inputs, which can confuse the model. 
    
    # Initialise a prompt specifying the amount of values that were non-empty inside of the batch to improve robustness
    prepared_prompt, idx = prepare_prompt_for_translation(batch["overleg"], batch_size=len(batch))
    
    attempt = 0
    while attempt < max_retries:
        try:
            completion_overleg = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                store=True,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": f"""
                        {prepared_prompt} 
                        Each entry is a field from a data frame column called 'overleg' describing college hours on particular dates in natural language. 

                        Your task is to return a JSON containing exactly {len(batch)} lists, where:
                        - Each employee gets a list of dictionaries and this list is allowed to be empty if this is an appropriate representation of the input.
                        - Each dictionary represents one date and contains:
                            - A key: the day of the month as a string (e.g. "14")
                            - A value: a string with the start and end hour in 24-hour format as integers (e.g. '8-13') 
                        - If the employee's input contains "[-]", return an empty list `[]` instead. Make sure that you return an empty list always if an input is lacking such that your output length always matches the input length. 
                        - If only a duration is mentiond (e.g.: '3 juli 8uur scholing'), then transform assign time window 06:00 to 23:30 as a default value, resulting in {{"3": "6-23.5"}}.
                        For example: 
                        {{
                        "Overleg": 
                            [{{"2": "14-16"]}}], # Employee has college on the second of the month between 14:00 and 16:00
                            [{{"14": "12-17"}}, {{"21": "15-17"]}}], # Employee has meeting or school on the 14th and 21st of the month between 12:00 and 17:00 and between 15:00 and 17:00 respectively
                            [] # Employee has no college this month
                            ... # until {len(batch)}-th list element.
                        }}
                        
                        IMPORTANT: Each employee must be represented as a list of dictionaries. Do not return a list of strings. Even for one day, wrap it like [{{"4": "6-23.5"}}] — not ["4", "6-23.5"].

                    """},
                    {"role": "user", "content": json.dumps(batch[['overleg']].fillna("[]").to_dict(orient='records'))}
                ],
                response_format=tool_overleg,
            )
                       
            # Step 1: Parse raw JSON string response
            response_json = json.loads(completion_overleg.choices[0].message.content)
            #print(response_json)
            overleg_tijden_batch = response_json["Overleg"]

            # Step 2: Validate + normalize structure
            validated_output = parse_and_validate_tijdwensen(overleg_tijden_batch, len(batch), idx)
            print(validated_output)
            overleg_tijden.extend(validated_output)
            time.sleep(1)  
            
            break # Move on after succesful batch
                    
        except Exception as e:
            attempt += 1
            print(f"Error occurred in batch {batch_start}-{batch_start+batch_size}: {e}. Attempt {attempt}/{max_retries}")
            print(e)
            time.sleep(2)

# If max retries reached, raise error
if attempt == max_retries:
    raise RuntimeError("Max retry attempts reached. Exiting.")

# Fix the output
import re
import ast

def fix_bracketed_list_string(text):
    """
    Cleans and fixes a malformed string that should represent a list of dictionaries.
    - Handles empty string variants like '', '""'.
    - Ensures matching brackets.
    - Trims characters outside the outermost list brackets.
    - Returns a parsed list or an empty list if parsing fails.
    """
    if not isinstance(text, str) or text.strip() in ["", "''", '""']:
        return []

    # Extract the outermost bracketed list from the string
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    bracketed_text = match.group(0)

    # Balance unclosed brackets
    open_brackets = bracketed_text.count("[")
    close_brackets = bracketed_text.count("]")

    if open_brackets > close_brackets:
        bracketed_text += "]" * (open_brackets - close_brackets)

    try:
        result = ast.literal_eval(bracketed_text)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# In[17]:


#### -------####


# #### Gebruik AI om de kolommen met tijdwensen om te zetten naar code-suited inputs

# In[18]:


### Retrieve the time preferences for employees
### Dit betreft indicaties van voorkeurstijden op bepaalde dagen
### De default logica zorgt ervoor 
attempt = 0

tijd_wensen = []  # Store results

for batch_start in range(0, len(emp_file), batch_size):
    batch = emp_file.iloc[batch_start : batch_start + batch_size,]
    print(batch[["tijd wensen"]].fillna("[]"))

    # Initialise a prompt specifying the amount of values that were non-empty inside of the batch to improve robustness
    prepared_prompt, idx = prepare_prompt_for_translation(batch["tijd wensen"], batch_size=len(batch))
    
    attempt = 0
    while attempt < max_retries:
        try:
            completion_tijd_wensen = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                store=True,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": f"""
                        {prepared_prompt} 
                        Each entry is a field from a data frame column called 'tijd wensen' describing time slots on weekdays that they prefer to be scheduled

                        Your task is to return a JSON containing exactly {len(batch)} lists, where:
                        - Each employee gets a list of dictionaries and this list is allowed to be empty if this is an appropriate representation of the input.
                        - Each dictionary represents a weekday and contains:
                            - A key: the day of the week (e.g. some number between 1 and 7. 1 = 'Maandag' and 7 = 'Zondag')
                            - A value: a string with the prefered time window in 24-hour format as integers (e.g. '8-13') 
                        - If the employee's input contains 'NaN' or is empty, return an empty list `[]`. It is possible that a batch contains only 'Nan' values. Make sure to return an empty list in such cases and make sure that the length of the input matches the length of your ouptut. 
                        - If only a duration is mentioned rather than a weekday and time (e.g.: 'Alleen van 15:00 tot 23:30'), then this naturally applies for all days of the week (hence this requires 7 dictionaries to represent each day separately).
                        - If a negative statement is shown, such as 'Donderdag niet van 18:00 tot 23:30', then the said time window needs to be excluded from the corresponding dictionary (but you will still have a dictionary for each day if an input is provided!!). 
                        - These negative statements can lead to preferred times being split in two parts throughout the day (e.g. when the input 'Niet van 10 tot 14'), such that there is a available time window 
                        - Sometimes time preferences are complemented with a phrase like 'niet beschibkaar op andere dagen', meaning that the other days have no available time slots. In these cases those other weekdays can be assigned a '0-0' time interval. 
                        bot before and after the provided time mentioned. In these cases, create additional dictionaries for those days such that all the remaining available windows are properly reprsented. See the example below for a clear example. 
                        
                        For your process, for each nonempty row in the input json batch, start with 
                        1. a complete seven day dictionary like: 
                        [{{"1": "0-23.75""]}}, {{"2": "0-23.75""]}}, {{"3": "0-23.75""]}}, {{"4": "0-23.75""]}}, {{"5": "0-23.75""]}}, {{"6": "0-23.75""]}}, {{"7": "0-23.75""]}}]
                        2. Read the input value inside the json for the particular row
                        3. Adjust the value of the item in the dictionary to match the time windows described in the text that was just read, taking into account the negative phrasing when this phrasing is applied. 
                        4. Advance to the next row of the json batch
                        
                        For example, your output should look like: 
                        {{
                        "tijd_wensen": 
                            [{{"1": "14-23.5"]}}, {{"2": "14-23.5"]}}, {{"3": "14-23.5"]}}, {{"4": "14-23.5"]}}, {{"5": "14-23.5"]}}, {{"6": "14-23.5"]}}, {{"7": "14-23.5"]}}], # Employee has preference 'Alleen van 14:00 tot 23:30')
                            [{{"1": "0-23.75"]}}, {{"2": "0-23.75"]}}, {{"3": "0-7}}, {{"3": "14-23.5"]}}, {{"4": "0-23.75"]}}, {{"5": "0-23.75"]}}, {{"6": ""0-23.75"]}}, {{"7": "0-23.75"]}}], # Employee has preference 'Woensdag niet tussen 07:00 en 14:00'. Note that we thus have two dictionaries for Wednesday in this case!
                            [{{"1": "0-23.75"]}}, {{"2": "0-23.75"]}}, {{"3": "0-15.5"]}}, {{"4": "0-23.75"]}}, {{"5": "0-23.75"]}}, {{"6": "0-15:30"]}}, {{"7": "0-23.75"]}}] # Employee has preference 'Woensdag en zaterdag niet na 15:30'. Note than only the dictionary with key "3" (for Wednesday) has an adjusted time window
                            [{{"1": "15-23.75"]}}, {{"2": "15-23.75""]}}, {{"3": "15-23.75"]}}, {{"4": "15-23.75"]}}, {{"5": "15-23.75"]}}, {{"6": "15-23.75"]}}, {{"7": "15-23.75"]}}] # Employee has preference 'Niet van 0.00 tot 15.00. Note only one dictionary per day is required because the preferred time implied splits the day in two parts (0-15 and 15-24)
                            [{{"1": "7-15.5"]}}, {{"2": "7-15.5"]}}, {{"3": "7-15.5"]}}, {{"4": "7-15.5"]}}, {{"5": "7-15.5"]}}, {{"6": "7-15.5"]}}, {{"7": "7-15.5"]}}] # Employee has preference 'Alleen van 7.00 tot 15:30. Note only one dictionary per day is required because the preferred time implied splits the day in two parts (0-15.5 and 16-24)
                            [{{"1": "7-10"]}}, {{"2": "0-0"]}}, {{"3": "7-10"]}}, {{"4": "0-0"]}}, {{"5": "0-10"]}}, {{"6": "0-0"]}}, {{"7": 0-0"]}}] # Employee has preference 'Maandag, woensdag en vrijdag van 7 tot 10. Op andere weekdagen niet beschibkaar'

                            ... # until {len(batch)}-th list element.
                        }}
                        
                        IMPORTANT: In the case of a phrase like "Niet van 07:00 tot 14:30", (note that no weekdays were mentioned) take the remaining available time window that exist after the given window for all of the weekdays. The latter example would hence result in "14.5-23.75" for ALL weekdays. 
                        IMPORTANT: Double check if you are applying the right time slots to the right days. 1 = maandag, 2 = dinsdag etc.. This is especially important when processing fields that contain phrases like "maandag en donderdag" (meaning weekdays 1 and 4) versus "maandag t/m donderdag" (meaning weekdays 1,2,3,4).
                        IMPORTANT: Each employee must be represented as a list of dictionaries. Do not return a list of strings.

                    """},
                    {"role": "user", "content": json.dumps(batch[['tijd wensen']].fillna("[]").to_dict(orient='records'))}
                ],
                response_format=tool_tijd,
            )

            # Step 1: Parse raw JSON string response
            response_json = json.loads(completion_tijd_wensen.choices[0].message.content)
            #print(response_json)
            tijd_wensen_batch = response_json["TijdWensen"]

            # Step 2: Validate + normalize structure
            validated_output = parse_and_validate_tijdwensen(tijd_wensen_batch, len(batch), idx)
            print(validated_output)
            tijd_wensen.extend(validated_output)
            time.sleep(1)
                
            break  # Success, move to next batch
            
        except Exception as e:
            attempt += 1
            print(f"Error occurred in batch {batch_start}-{batch_start+batch_size}: {e}. Attempt {attempt}/{max_retries}")
            print(e)
            time.sleep(2)

# If max retries reached, raise error
if attempt == max_retries:
    raise RuntimeError("Max retry attempts reached. Exiting.")


# #### Verwerken van de scholing en tijd kolom

# In[19]:


### Retrieve the scholingstijden for the employees
### De default logica zorgt ervoor dat bij 8 urige schooldagen de alle shifts worden geblokkeerd doordat er geen shifts zijn die op die dag kunnen beginnen of eindigen tussen 6 and 17 uur. 
attempt = 0
scholing_tijden = []  # Store results

for batch_start in range(0, len(emp_file), batch_size):
    batch = emp_file.iloc[batch_start : batch_start + batch_size,]
    print(batch[["scholing en tijd"]].fillna("[]")) # add a dash to prevent truly empty inputs, which can confuse the model. 
    
    # Initialise a prompt specifying the amount of values that were non-empty inside of the batch to improve robustness
    prepared_prompt, idx = prepare_prompt_for_translation(batch["scholing en tijd"], batch_size=len(batch))
    
    attempt = 0
    while attempt < max_retries:
        try:
            completion_scholing = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                store=True,
                timeout=timeout,
                messages=[
                    {"role": "system", "content": f"""
                        {prepared_prompt} 
                        Each entry is a field from a data frame column called 'scholing en tijd' describing college hours on particular dates in natural language. 

                        Your task is to return a JSON containing exactly {len(batch)} lists, where:
                        - Each employee gets a list of dictionaries and this list is allowed to be empty if this is an appropriate representation of the input.
                        - Each dictionary in the list represents one date and contains:
                            - A key: the day of the month as a string (e.g. "14")
                            - A value: a string with the start and end hour in 24-hour format as integers (e.g. '8-13') 
                        - If the employee's input contains "[-]", return an empty list `[]` instead. Make sure that you return an empty list always if an input is lacking such that your output length always matches the input length. 
                        - If only a duration is mentiond (e.g.: '3 juli 8uur scholing'), then transform assign time window 06:00 to 23:30 as a default value, resulting in {{"3": "6-23.5"}}.
                        - If a range of dates is specified, make sure to return a dictionary for each day in this range.
                        For example: 
                        {{
                        "Scholing": 
                            [{{"2": "14-16"]}}], # Employee has college on the second of the month between 14:00 and 16:00
                            [{{"14": "12-17"}}, {{"21": "15-17"]}}], # Employee has meeting or school on the 14th and 21st of the month between 12:00 and 17:00 and between 15:00 and 17:00 respectively
                            [{{"15": "12-17"}}, {{"16": "15-17"]}}, {{"17": "15-17"]}},  {{"18": "15-17"]}}], # Employee's input was '15 tm 18 van 15:00 tot 17:00'
                            [] # Employee has no college this month
                            ... # until {len(batch)}-th list element.
                        }}
                        
                        IMPORTANT: Each employee must be represented as a list of dictionaries. Do not return a list of strings. Even for one day, wrap it like [{{"4": "6-23.5"}}] — not ["4", "6-23.5"].

                    """},
                    {"role": "user", "content": json.dumps(batch[['scholing en tijd']].fillna("[]").to_dict(orient='records'))}
                ],
                response_format=tool_scholing,
            )
           
            # Step 1: Parse raw JSON string response
            response_json = json.loads(completion_scholing.choices[0].message.content)
            #print(response_json)
            scholing_batch = response_json["Scholing"]

            # Step 2: Validate + normalize structure
            validated_output = parse_and_validate_tijdwensen(scholing_batch, len(batch), idx)
            print(validated_output)
            scholing_tijden.extend(validated_output)
            time.sleep(1)
                
            break  # Success, move to next batch
            
                    
        except Exception as e:
            attempt += 1
            print(f"Error occurred in batch {batch_start}-{batch_start+batch_size}: {e}. Attempt {attempt}/{max_retries}")
            print(e)
            time.sleep(2)

# If max retries reached, raise error
if attempt == max_retries:
    raise RuntimeError("Max retry attempts reached. Exiting.")


# #### Opschonen van AI output en de waarden aan de tabel koppelen

# In[ ]:


# Fix any malformed outputs output
import re
import ast

def fix_bracketed_list_string(text):
    """
    Cleans and fixes a malformed string that should represent a list of dictionaries.
    - Handles empty string variants like '', '""'.
    - Ensures matching brackets.
    - Trims characters outside the outermost list brackets.
    - Returns a parsed list or an empty list if parsing fails.
    """
    if not isinstance(text, str) or text.strip() in ["", "''", '""']:
        return []

    # Extract the outermost bracketed list from the string
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    bracketed_text = match.group(0)

    # Balance unclosed brackets
    open_brackets = bracketed_text.count("[")
    close_brackets = bracketed_text.count("]")

    if open_brackets > close_brackets:
        bracketed_text += "]" * (open_brackets - close_brackets)

    try:
        result = ast.literal_eval(bracketed_text)
        return result if isinstance(result, list) else []
    except Exception:
        return []

def flatten_if_needed(value):
    # Only flatten if it's a list with a single list inside
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]  # Unpack the inner list
    return value  # Return as-is

emp_file["overleg_tijden"] = overleg_tijden  
emp_file["overleg_tijden"] = emp_file["overleg_tijden"].apply(fix_bracketed_list_string)
emp_file["tijd_wensen"] = tijd_wensen
emp_file["scholing_tijden"] = scholing_tijden
emp_file["vrije_dagen"] = all_wensen_vrij
emp_file["vrije_dagen"] = emp_file["vrije_dagen"].apply(flatten_if_needed)

print(emp_file["scholing_tijden"].iloc[40:80])
print(emp_file["vrije_dagen"].iloc[20:50])
print(emp_file["tijd_wensen"].iloc[70:90])


# ## Verwerken van resterende data t.b.v. rooster tool

# ##### Specificeer inputs

# In[ ]:


contract_naar_maand_factor = 5.5 # Om maandelijkse uren te berekenen
max_hour_limit = 1.05 # Bepaalt hoeveel uren er overgewerkt mag worden op maandbasis


# #### Zet datumfuncties klaar om weeknummers, weekdagnummer en aantal dagen in de maand op te halen

# In[ ]:


# Set up a calendar
from datetime import date, datetime, timedelta
import calendar

def get_week_number(year, month, day):
    """ Geeft het weeknummer van een datum in het jaar terug (1-52). """
    return datetime(year, month, day).isocalendar()[1]

def get_weekday(year, month, day_index):
    return (datetime.date(year, month, day_index + 1).weekday() + 1)  # Monday = 1, Sunday = 7


def get_days_in_month(year: int, month: int) -> int:
    """Returns the number of days in a given month and year."""
    return calendar.monthrange(year, month)[1] 

print(get_days_in_month(REFERENCE_YEAR, REFERENCE_MONTH))


# #### Functies voor databewerking
# ##### Standaardoperaties

# In[ ]:


import re
import ast
import numpy as np

# Create sample contract hours

# Create branch id's and mapping
unique_branches = emp_file["filiaal"].dropna().unique()
branch_id_map = {name: idx for idx, name in enumerate(unique_branches, start= 0)}
ID_TO_BRANCH = {v: k for k, v in branch_id_map.items()}

print("📌 Branch ID Mapping:", branch_id_map)  # Debugging

def parse_additional_branches(entry, all_branches):
    entry = entry.lower()
    if "alles" in entry or "overal" in entry:
        # Start with all cities
        branches = set(all_branches)
        
        # Check for 'except'
        if "behalve" in entry:
            for branch in all_branches:
                if branch.lower() in entry:
                    branches.discard(branch)
        return list(branches)
    else:
        return entry


def clean_branches(branch_str):
    if branch_str is None or branch_str == '' or (isinstance(branch_str, float) and pd.isna(branch_str)):
        return []

    # If already a list, use it directly
    if isinstance(branch_str, list):
        branch_list = branch_str
    else:
        branch_list = str(branch_str).strip().split(",")

    # Clean each item
    branch_list = [re.sub(r'\bja\b', '', item, flags=re.IGNORECASE) for item in branch_list]
    cleaned_branches = [re.sub(r'\s+', '', name).lower() for name in branch_list]
    cleaned_branches = [name for name in cleaned_branches if name]

    return cleaned_branches


def map_branch_names_to_ids(branch_str):
    # Handle missing values
    if branch_str is None or branch_str == '' or (isinstance(branch_str, float) and pd.isna(branch_str)):
        return []

    # Clean names (string or list compatible)
    cleaned_branches = clean_branches(branch_str)

    # Map cleaned names to IDs
    branch_ids = [branch_id_map[name] for name in cleaned_branches if name in branch_id_map]

    return branch_ids


def merge_branch_columns(row):
    # Extract lists from both columns (default to empty lists if NaN)
    primary_branches = row["branch_ids"] if isinstance(row["branch_ids"], list) else []
    extra_branches = row["branches_extra"] if isinstance(row["branches_extra"], list) else []

    # ✅ Combine both lists and remove duplicates
    combined_branches = list(set(primary_branches + extra_branches))
    
    return combined_branches


# Create binary column for night shifts for more efficient solving
def night_shift_indicator(row):
    if isinstance(row["werkt nachtdienst"], str):
        return True
    else:
        return False

def extract_days_from_string(entry: str) -> list:
    if pd.isna(entry):
        return []
    return list(map(int, re.findall(r'\d+', entry)))


    
def get_blacklisted_days_by_month(year: int, month: int, preference: str):
    if pd.isna(preference) or pd.isna(year) or pd.isna(month):
        return []

    # Normalize input
    preference = preference.lower()

    # Week parity logic
    blacklist_odd = re.search(r'\boneven\b', preference)
    blacklist_even = re.search(r'\beven\b', preference)

    if not (blacklist_odd or blacklist_even):
        print(preference)
        raise ValueError("Preference must mention either 'odd' or 'even' weeks.")

    # Weekday name to index mapping
    weekday_map = {
        'maandag': 0, 'dinsdag': 1, 'woensdag': 2, 'donderdag': 3,
        'vrijdag': 4, 'zaterdag': 5, 'zondag': 6
    }

    # Extract mentioned weekdays
    selected_weekdays = set()

    # Check for specific weekday mentions
    for name, idx in weekday_map.items():
        if name in preference:
            selected_weekdays.add(idx)

    # Add weekend days if mentioned
    if "weekend" in preference:
        selected_weekdays.update([5, 6])  # Saturday and Sunday
        if not selected_weekdays:
            raise ValueError("No valid weekday names found in preference string.")

    # Define date range for the month
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])
    delta = timedelta(days=1)

    blacklisted_dates = []

    current = start_date
    while current <= end_date:
        weekday = current.weekday()
        week_number = current.isocalendar().week

        if weekday in selected_weekdays:
            if (blacklist_odd and week_number % 2 == 0) or (blacklist_even and week_number % 2 == 1): # Use mod 0 for odd and mod 1 for even to invert the weeks in which the wish is applied since the preferred days are mentioned, and the opposites need to be blacklisted. 
                blacklisted_dates.append(current.day)

        current += delta

    return sorted(blacklisted_dates)


# Function to merge rows by keeping the most informative values
def merge_most_informative(group):
    
    for col in group.columns:
        if group[col].dtype == "object":
            group[col] = group[col].apply(lambda x: x.lower() if isinstance(x, str) else x)

    merged_row = {}
    for col in group.columns:
        if col == "naam":
            merged_row[col] = group[col].iloc[0]  # Keep name as it is
        elif group[col].dtype == "int64" or group[col].dtype == "float64":
            merged_row[col] = group[col].max()  # Keep the max numerical value
        elif group[col].apply(lambda x: isinstance(x, list)).all():
            merged_row[col] = max(group[col], key=len, default=[])  # Keep the longest list
        elif group[col].apply(lambda x: isinstance(x, str)).all():
            merged_row[col] = max(group[col], key=len, default="")  # Keep the longest string
        else:
            merged_row[col] = group[col].iloc[0]  # Default fallback
    return pd.Series(merged_row)


# Append vrije/vakantiedagen
def clean_vacation_days(vacation_list, year, month):
    """Clean a list of vacation days to include only valid integers within month range."""
    max_days = calendar.monthrange(year, month)[1]
    if not isinstance(vacation_list, list):
        return []
    return [
        int(d) for d in vacation_list
        if isinstance(d, (int, float)) and 1 <= int(d) <= max_days
    ]


# Functie om verdeelseutel op te schonen
def parse_branch_distribution(dist_str):
    if not dist_str or pd.isna(dist_str):
        return {}  # No preference

    try:
        raw_pairs = dist_str.split(',')
        branch_weights = {}
        total = 0

        for pair in raw_pairs:
            branch_name, weight = pair.strip().split(':')
            branch_id   = map_branch_names_to_ids(branch_name) # cleans and maps
            branch_id = branch_id[0]
            print(branch_name, branch_id)
            weight = float(weight)        

            if weight < 1:
                weight = weight * 100
            #print(branch_id, weight)
            branch_weights[branch_id] = weight
            print(branch_weights)
            total += weight

            
        # Normalize
        #if total > 0:
        #    for key in branch_weights:
        #        branch_weights[key] /= total
        #        print(branch_weights[key])
        print(branch_weights)
        return branch_weights

    except Exception as e:
        print(f"⚠️ Could not parse branch_distribution: {dist_str} — {e}")
        return {}



# Functie om de verdeelsleutel van de beschikbare branches goed te verwerken tot data
def complete_branch_distribution(allowed_branches, distribution_dict):
    """
    Completes and normalizes a shift distribution across allowed branches for an employee.

    Args:
        allowed_branches (list): Branch IDs the employee is allowed to work at.
        distribution_dict (dict): Partial branch distribution, e.g., {0: 40, 2: 30}

    Returns:
        tuple: (complete_distribution: dict, distribution_specified: bool)
    """
    # No preference provided → no constraint needed
    if not distribution_dict:
        return {}, False

    # Ensure all branch keys are integers
    distribution_dict = {int(k): v for k, v in distribution_dict.items()}

    specified_total = sum(distribution_dict.values())
    unspecified_branches = [b for b in allowed_branches if b not in distribution_dict]

    complete_distribution = distribution_dict.copy()

    if unspecified_branches and specified_total < 100:
        # Distribute remaining percentage across unspecified branches
        remaining = 100 - specified_total
        per_branch = remaining / len(unspecified_branches)

        for b in unspecified_branches:
            complete_distribution[b] = round(per_branch)

        # Correct rounding error
        total_now = sum(complete_distribution.values())
        diff = 100 - total_now
        if diff != 0:
            first = next(iter(complete_distribution))
            complete_distribution[first] += diff

    else:
        # Either all branches are specified OR sum is too high → normalize
        scale = 100 / specified_total if specified_total != 0 else 0
        complete_distribution = {
            b: round(distribution_dict.get(b, 0) * scale) for b in allowed_branches
        }

        # Correct rounding
        total_now = sum(complete_distribution.values())
        diff = 100 - total_now
        print("diff", diff)
        if diff != 0:
            first = next(iter(complete_distribution))
            complete_distribution[first] += diff
            print(allowed_branches)
            

    return complete_distribution, True


# Define meta function for computing the overall disribution of the preferred branches
def process_distribution_row(row):
    parsed = parse_branch_distribution(row["verdeelsleutel"])
    allowed = row["all_branch_ids"]
    return complete_branch_distribution(allowed, parsed)


# Define function to more strictly define overlap between preferred times and shift times
def shifts_in_preferred_time(shift_start, shift_end, pref_start, pref_end):
    # Normalize for overnight shifts
    if shift_end <= shift_start:
        shift_end += 24
    if pref_end <= pref_start:
        pref_end += 24

    return shift_start >= pref_start and shift_end <= pref_end


def compute_net_hours(row):
    contract_hours = row['contracturen']
    vacation_days = row['vakantie']
    
    # Count vacation days that fall on weekdays
    weekday_vac_days = [
        d for d in vacation_days
        if date(REFERENCE_YEAR, REFERENCE_MONTH, d).weekday() < 5
    ]
    
    vacation_ratio = len(weekday_vac_days) / get_days_in_month(REFERENCE_YEAR, REFERENCE_MONTH) 
    vacation_hours = contract_hours * vacation_ratio # compare the amount of vacation days to the amount of days in the month
    net_hours = max(contract_hours - vacation_hours, 0) # Sometimes this appears to be negative, how can this be the case?
    
    verlofuren = contract_hours / contract_naar_maand_factor / 5 * len(weekday_vac_days) # is verlof per dag (in werkweek: dus delen door 5) x aantal vakantiedagen
    net_hours = max(contract_hours - verlofuren, 0)
    print(f'contracturen: {contract_hours}', f' netto uren: {net_hours}', f' aantal vakantiedagen: {len(weekday_vac_days)}', f' fte ratio: {vacation_ratio}', f' vacation hours: {vacation_hours}')

    return round(net_hours, 2)


# #### Pas de functies toe

# In[ ]:


# Set up the branch ID's
emp_file["branch_ids"]     = emp_file["filiaal"].apply(map_branch_names_to_ids) # assign id's to branches
all_branches_list = emp_file["filiaal"].unique().tolist() # get all branch names
emp_file["bedient andere afd2"] = emp_file["bedient andere afd"].apply(lambda x: parse_additional_branches(str(x), all_branches_list)) # .fillna("").apply(clean_branches) # clean branches for 'bedient andere afd' column
emp_file["branches_extra"] = emp_file["bedient andere afd2"]
emp_file["branches_extra"] = emp_file["branches_extra"].apply(map_branch_names_to_ids) # assign id for every other branch mentioned in 'bedient andere afd' column
emp_file["all_branch_ids"] = emp_file.apply(merge_branch_columns, axis=1) # combine main and additional branches into one column

# ConProcessvert the night shift indicator column
emp_file["works_night_shifts"] = emp_file.apply(night_shift_indicator, axis = 1)

# Correct the hours in case of empty entries
emp_file["contract uren"] = emp_file["contract uren"].fillna(0)

# Convert comma-decimal format (e.g., 20,5 → 20.5)
emp_file["contracturen"] = (
    pd.to_numeric(emp_file["contract uren"], errors="coerce")  # convert to numeric, NaNs if invalid
    .fillna(0)                                               # replace NaNs with 0
    .clip(lower=0)    
    .astype(str)          # Ensure all values are strings
    .str.replace(",", ".")  # Replace commas with dots
    .astype(float)        # Convert to float
)

# Append hours column to emp_file and adjust for structural leave hours (maternal leave, cleaning tasks etc.)
emp_file["contracturen"] = (emp_file["contract uren"] - emp_file["alternatief verlof"].fillna(0)) * contract_naar_maand_factor  # dit moet 4.5 worden i.v.m. 


# Set up priority list for consistent scheduling
emp_file["prioriteit"] = emp_file.shape[0] - np.arange(emp_file.shape[0]) # inverteer de prioritering => hoge getallen krijgen eerste behandeling

# Clean vacation days
emp_file["vakantie"] =  [
    clean_vacation_days(sublist, REFERENCE_YEAR, REFERENCE_MONTH)
    for sublist in all_vacation_days # iterate over GPT interpreted vacation days
]

# assign special leave days (both official and unofficial leave (niet verlof))
emp_file['vrije_dagen_nv'] = all_vrije_nv_days

#emp_file["vrije_dagen"] = all_wensen_vrij # assign GPT interpreted leave days

## Process the verdeelsleutel column
emp_file[["branch_distribution", "has_distribution"]] = emp_file.apply(
    lambda row: pd.Series(process_distribution_row(row)), axis=1
)

# Create functie id 
emp_file["functie"] = [func.lower().strip() for func in emp_file["functie"]] # Clean the column
function_names = emp_file["functie"].unique() # extract unique names
FUNCTION_ID_MAP = {'bbl helpende': 0, 'bbl verz ig': 1, 'bbl vpk': 2, 'helpende': 3, 'helpende plus': 4, 'verz ig': 5, 'vpk': 6, 'vpk in opl': 7, 'woonass': 8} # assign a fixed mapping (because of manually specified logic later)
#

ID_TO_FUNCTION = {v: k for k, v in FUNCTION_ID_MAP.items()}
# Map the function names to a function id
emp_file["functie_id"] = emp_file["functie"].map(FUNCTION_ID_MAP)

# Process the odd/even preferences for each employee
emp_file['blacklisted_dates'] = emp_file.apply(
    lambda row: get_blacklisted_days_by_month(REFERENCE_YEAR, REFERENCE_MONTH, row['(on)even wensen']),
    axis=1
)

# Introduce new object for consistency purposes and simplicity
df = emp_file

# Merge duplicates based on the 'name' column. Coalesce values for similar names based on the most informative value between duplicate name rows
print("shape: " , df.shape)
print(len(df["naam"].unique()))
df = df.groupby("naam", group_keys=False).apply(merge_most_informative).reset_index(drop=True)
print("shape: " , df.shape)
print(len(df["naam"].unique()))

df['net_contract_uren'] = df.apply(compute_net_hours, axis=1)

df['evv'] = df['evv'].astype(str)


# #### Employee object aanmaken en populaten

# In[ ]:


# Klasse om een medewerker te representeren
class Employee:
    # Definieer de klasse
    def __init__(self
                 , name                 = None  # naam
                 , allowed_branches     = None  # toegestane afdelingen
                 , contract_hours       = 0     # contracturen per maand
                 , net_contract_hours   = 0     # contracturen minus verlofuren
                 , vacation             = None  # vakantiedagen
                 , free_days            = None  # vrije dagen (niet verlof)
                 , unavailable_days     = None  # niet beschikbare dagen (harde wens)
                 , blacklisted_days     = None  # vrije dagen (oneven/even vrije dagen)
                 , blocked_hours        = None  # tijden voor scholing
                 , preferred_times      = None  # Tijden waarop iemand wil/kan werken
                 , branch_distribution  = None  # Verdeelsleutel
                 , priority             = 1     # prioriteit (voor het roostermaken)
                 , works_night_shifts   = 0     # of nachtshifts zijn toegestaan
                 , func                 = 0     # de funcite id van de medewerker
                 , contract_type        = None  # omschrijving dienstverband (vast of flex)
                 , meeting_times        = None  # lijst van tijden waarop iemand overleg heeft en een bijpassende shift verlangt
                 , is_evv               = None
                 ):
        
        # Vul de velden van de klasse in o.b.v. de inputs
        self.name = name
        self.allowed_branches = allowed_branches if isinstance(allowed_branches, list) else [allowed_branches]
        self.branch_distribution = branch_distribution or {}
        self.contract_hours = contract_hours
        self.net_contract_hours = net_contract_hours
        self.unavailable_days = unavailable_days or []  # Lijst van dagen waarop niet gewerkt kan worden (bijv. [5, 6] voor weekend)
        self.blacklisted_days = blacklisted_days or []
        self.blocked_hours    = blocked_hours or []
        self.preferred_times  = preferred_times or []
        self.works_night_shifts = works_night_shifts
        self.assigned_hours = 0
        self.priority = priority
        self.vacation = vacation or []
        self.free_days = free_days or []
        self.func = func 
        self.contract_type = contract_type
        self.meeting_times = meeting_times
        self.is_evv = is_evv
        
    def __repr__(self):
        return f"Employee({self.name}, {self.allowed_branches}, {self.contract_hours})"
        # Days: {self.unavailable_days}, Shifts: {self.unavailable_shifts})
        

# Verwerken van data naar Employee objecten
employees = []
for _, row in df.iterrows():
    name                = row["naam"]
    allowed_branches    = row["all_branch_ids"]
    branch_distribution = row["branch_distribution"]
    contract_hours      = row["contracturen"]
    net_contract_hours  = row["net_contract_uren"]
    vacation            = row["vakantie"]
    free_days           = row["vrije_dagen_nv"]
    unavailable_days    = row["vrije_dagen"]
    blacklisted_days    = row["blacklisted_dates"]
    blocked_hours       = row["scholing_tijden"]  # was overleg_tijden
    preffered_times     = row["tijd_wensen"]
    prioriteit          = row["prioriteit"]
    works_night_shifts  = bool(row["works_night_shifts"])
    func                = row["functie_id"]
    contract_type       = row["contracttype"].lower()
    meeting_times       = row["overleg_tijden"]
    is_evv              = row['evv']

    # Maak een Employee object en voeg toe aan lijst
    employees.append(Employee(name, 
                              allowed_branches=allowed_branches,
                              branch_distribution=branch_distribution,
                              contract_hours= contract_hours, 
                              net_contract_hours=net_contract_hours,
                              vacation =vacation, 
                              free_days=free_days,
                              unavailable_days = unavailable_days, # These are the fixed free days for each week # TODO: verkeerd?
                              blacklisted_days = blacklisted_days, # These are the even/odd blocked days in this particular case
                              blocked_hours    = blocked_hours,    # These are the hours that an employee has meetings, or classes for related education
                              preferred_times  = preffered_times,
                              priority=prioriteit, 
                              works_night_shifts=works_night_shifts,
                              func = func,
                              contract_type = contract_type,
                              meeting_times = meeting_times,
                              is_evv = is_evv
                        
                              ))
    
if employees[0].unavailable_days:
    print('w')
else:
    print('l')


# In[ ]:


#employees


# In[ ]:


import numpy as np
# Klasse om een shift te representeren
class Shift:
    def __init__(self, shift_id, start_time, duration, worked_hours, allowed_functions=None, naam = "Geen naam", presence_mandatory =  False):
        self.shift_id = shift_id  # Bijvoorbeeld 0=ochtend, 1=middag, 2=avond
        self.start_time = start_time  # Starttijd in 24-uurs notatie (bijv. 6, 14, 22)
        self.duration = duration  # Hoe lang de shift duurt (bijv. 8 of 6 uur)
        self.worked_hours = worked_hours
        self.allowed_functions = allowed_functions or []
        self.naam = naam
        self.presence_mandatory = presence_mandatory
        
# Tijdstippen waarop aanwezigheid wordt gecontroleerd
TIME_SLOTS = np.arange(0, 24, 0.25)  # Elke 30 minuten (0, 0.5, 1, ..., 23.5)

# 0: West1, 1: West2, 2: West3, 3: zuid, 4: Oost1, 5: Oost2, 6: Oost3, 7: aanleun
SHIFT_PERIODS_PER_BRANCH = {
    "0": [  # West1
        {"name": "Ochtend1", "start": 7.0  , "end": 7.5  , "min_staff": 1},
        {"name": "Ochtend2", "start": 7.5  , "end": 8    , "min_staff": 2},
        {"name": "Ochtend3", "start": 8.0  , "end": 12.5 , "min_staff": 3},
        {"name": "Middag1" , "start": 12.5 , "end": 13   , "min_staff": 2},
        {"name": "Middag2" , "start": 13.0 , "end": 14   , "min_staff": 1},
        {"name": "Middag3" , "start": 14.0 , "end": 15.25, "min_staff": 2},
        {"name": "Avond1"  , "start": 15.25, "end": 15.5 , "min_staff": 3},
        {"name": "Avond2"  , "start": 15.5 , "end": 22.5 , "min_staff": 2},
        {"name": "Avond3"  , "start": 22.5 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 1},  # Nacht loopt over middernacht heen!
    ],
    "1": [ # West2
        {"name": "Ochtend1", "start": 7.0  , "end": 8.5  , "min_staff": 2},
        {"name": "Ochtend2", "start": 8.5  , "end": 12.0 , "min_staff": 3},
        {"name": "Ochtend3", "start": 12.0 , "end": 13.0 , "min_staff": 2},
        {"name": "Middag1" , "start": 13.0 , "end": 13.5 , "min_staff": 1},
        {"name": "middag2" , "start": 13.5 , "end": 15.25, "min_staff": 2},
        {"name": "middag3" , "start": 15.25, "end": 15.5 , "min_staff": 3},
        {"name": "avond1"  , "start": 15.5 , "end": 22.0 , "min_staff": 2},
        {"name": "avond2"  , "start": 22.0 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 1},  # Nacht loopt over middernacht heen!
    ],
    "2": [ # West3
        {"name": "Ochtend1", "start": 7.0  , "end": 7.5  , "min_staff": 3},
        {"name": "Ochtend2", "start": 7.5  , "end": 10.0 , "min_staff": 4},
        {"name": "Ochtend3", "start": 10.0 , "end": 12.0 , "min_staff": 3},
        {"name": "Middag1" , "start": 12.0 , "end": 15.0 , "min_staff": 2},
        {"name": "Middag2" , "start": 15.0 , "end": 15.25, "min_staff": 1},
        {"name": "Middag3" , "start": 15.25, "end": 15.5 , "min_staff": 1}, # was 2
        {"name": "Middag4" , "start": 15.5 , "end": 16.0 , "min_staff": 1},
        {"name": "Middag5" , "start": 16.0 , "end": 21.0 , "min_staff": 2},
        {"name": "Avond"   , "start": 21.0 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 1},  # Nacht loopt over middernacht heen!
    ],
    "3": [ # Zuid
        {"name": "Ochtend1", "start": 7.0  , "end": 8.0  , "min_staff": 2},
        {"name": "Ochtend2", "start": 8.0  , "end": 13.0 , "min_staff": 3},
        {"name": "Middag1" , "start": 13.0 , "end": 13.5 , "min_staff": 2},
        {"name": "Middag2" , "start": 13.5 , "end": 14.0 , "min_staff": 1},
        {"name": "Middag3" , "start": 14.0 , "end": 15.25, "min_staff": 2},
        {"name": "Middag4" , "start": 15.25, "end": 15.5 , "min_staff": 3},
        {"name": "Avond1"  , "start": 15.5 , "end": 22.0 , "min_staff": 2},
        {"name": "Avond2"  , "start": 22.0 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 0}, # Nacht loopt over middernacht heen!
    ],
    "4": [ # Oost1
        {"name": "Ochtend1", "start": 7.0  , "end": 08.25, "min_staff": 2},
        {"name": "Ochtend2", "start": 8.25 , "end": 12.0 , "min_staff": 3},
        {"name": "Middag1" , "start": 12.0 , "end": 12.75, "min_staff": 2},
        {"name": "Middag2" , "start": 12.75, "end": 13.5 , "min_staff": 1},
        {"name": "Middag3" , "start": 13.5 , "end": 15.25, "min_staff": 2},
        {"name": "Middag4" , "start": 15.25, "end": 15.5 , "min_staff": 3},
        {"name": "Avond1"  , "start": 15.5 , "end": 22.0 , "min_staff": 2},
        {"name": "Avond2"  , "start": 22.0 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 1},  # Nacht loopt over middernacht heen!
    ],
    "5": [ # Oost2
        {"name": "Ochtend1", "start": 7.0  , "end": 7.5  , "min_staff": 1},
        {"name": "Ochtend2", "start": 7.5  , "end": 8.5  , "min_staff": 2},
        {"name": "Ochtend3", "start": 8.5  , "end": 12.0 , "min_staff": 3},
        {"name": "Ochtend4", "start": 12.0 , "end": 13.0 , "min_staff": 2},
        {"name": "Middag1" , "start": 13.0 , "end": 13.75, "min_staff": 1},
        {"name": "Middag2" , "start": 13.75, "end": 15.25, "min_staff": 2},
        {"name": "Middag3" , "start": 15.25, "end": 15.5 , "min_staff": 3},
        {"name": "Avond1"  , "start": 15.5 , "end": 22.25, "min_staff": 2},
        {"name": "Avond2"  , "start": 22.25, "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 1},  # Nacht loopt over middernacht heen!
    ],
    "6": [ # Oost3
        {"name": "Ochtend1", "start": 7.0  , "end": 7.5  , "min_staff": 4}, # 4 mensen i.v.m. Lutske die 6 uur werkt
        {"name": "Ochtend2", "start": 7.5  , "end": 10.0 , "min_staff": 5},
        {"name": "Ochtend3", "start": 10.0 , "end": 12.5 , "min_staff": 4},
        {"name": "Middag1" , "start": 12.5 , "end": 13.0 , "min_staff": 3},
        {"name": "Middag2" , "start": 13.0 , "end": 15.25, "min_staff": 2},
        {"name": "Middag3" , "start": 15.25, "end": 15.5 , "min_staff": 3},
        {"name": "Middag4" , "start": 15.5 , "end": 16.0 , "min_staff": 1},
        {"name": "Middag5" , "start": 16.0 , "end": 17.0 , "min_staff": 2},
        {"name": "Avond1"  , "start": 17.0 , "end": 20.0 , "min_staff": 3},
        {"name": "Avond2"  , "start": 20.0 , "end": 22.0 , "min_staff": 2},
        {"name": "Avond3"  , "start": 22.0 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 1},  # Nacht loopt over middernacht heen!
    ],
    "7": [  # Aanleun
        {"name": "Ochtend1", "start": 7.0  , "end": 7.5  , "min_staff": 1},
        {"name": "Ochtend2", "start": 7.5  , "end": 12.5 , "min_staff": 2},
        {"name": "Middag1" , "start": 12.5 , "end": 15.25, "min_staff": 1},
        {"name": "Middag2" , "start": 15.25, "end": 15.5 , "min_staff": 2},
        {"name": "Avond"   , "start": 15.5 , "end": 23.25, "min_staff": 1},
        {"name": "Nacht"   , "start": 23.25, "end": 7    , "min_staff": 0},  # Nacht loopt over middernacht heen!
    ]
}

WEEKS_PER_MONTH = 5  # Assume roughly 5 weeks per month. The 5th week is capped by the final day of the month. 

# Beschikbare shift-types (ID, starttijd, duur in uren)
#{'bbl helpende': 0, 'bbl verz ig': 1, 'bbl vpk': 2, 'helpende': 3, 'helpende plus': 4, 'verz ig': 5, 'vpk': 6, 'vpk in opl': 7, 'woonass': 8}
SHIFT_TYPES = {
    # 4 en 7 zijn verwijderd. 
    "0": [ # West1
        Shift(0, 7.0, 8.5, 8,        allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7.5, 5.0, 5,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst", presence_mandatory=True),
        Shift(3, 8.0, 5.0, 5,        allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True), # TODO: is functie 7 wel nodig op deze shifts
        Shift(4, 14.0, 8.5, 8,       allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(5, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),
    ],
    "1": [ # West2
        Shift(0, 7.0, 8.5, 8,        allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7.0, 5.0, 5,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(3, 8.5, 4.5,  4.5,     allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(4, 13.5, 8.5,  8.5,    allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst"),
        Shift(5, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),     
    ], 
    "2": [ # West3
        Shift(0, 7.0, 8.5,  8,       allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7.0, 3.0,   3,      allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(3, 7.0, 5.0,     5,    allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(4, 7.5, 7.5,   7,      allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(5, 16.0, 5.0, 5,       allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(6, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),
    ],
    "3": [ # Zuid
        Shift(0, 7.0, 8.5,  8,       allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7, 6,    6,         allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(3, 8.0, 5.5, 5.5,      allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(4, 14, 8, 8,           allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(5, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"), # is deze wel nodig?
    ],
    "4": [ # Oost1
        Shift(0, 7.0, 8.5, 8,        allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7, 5, 5,            allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst", presence_mandatory=True),
        Shift(3, 8.25, 4.5, 4.5,     allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(4, 13.5, 8.5, 8,       allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(5, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),
    ],
    "5": [ # Oost2
        Shift(0, 7.0, 8.5, 8,        allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7.5, 5.0, 5,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(3, 8.5, 4.5, 4.5,      allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(4, 13.75, 8.5, 8,      allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(5, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),
    ],
    "6": [ # Oost3
        Shift(0, 7.0, 8.5, 8,        allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 7.0, 8.5, 8,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"), # moet presence
        Shift(2, 7.0, 5.0, 5,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende plus dienst"),
        Shift(3, 7.0, 3.0, 3,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(4, 7.5, 5.5, 5.5,      allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True), # Dienst voor Lutske van 7 tot 9 wordt hier niet meegenomen. We hebben de uren van Lutske aangepast zodat deze passen bij 3 uur shifts (al werkt ze in realiteit maar 2 uur)
        Shift(5, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(6, 17.0, 5.0, 5,       allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(7, 16.0, 4.0, 4,       allowed_functions=[0,1,2,3,4,8], naam = "Woonass dienst", presence_mandatory=True),
        Shift(8, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),
    ],
    "7": [ # Aanleun
        Shift(0, 7.0, 8.5, 8,        allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(1, 15.25, 8.25, 8.25,  allowed_functions=[5,6,7], naam = "Verantw dienst", presence_mandatory=True),
        Shift(2, 7.5, 5.0, 5,        allowed_functions=[0,1,2,3,4,5,6,7], naam = "Helpende dienst"),
        Shift(3, 23.25, 8.25, 8.25,  allowed_functions=[0,1,2,3,4,5,6,7], naam = "Nacht dienst"),
    ],
}

# Map the branches to parts of the building for night shift purposes
BRANCH_TO_BUILDING = {
    "0": "Toren_west", "1": "Toren_west", 
    "2": "Toren_west", "3": "Zuid",
    "4": "Toren_oost", "5": "Toren_oost", 
    "6": "Toren_oost", "7": "Aanleun"
}

# Create a building oriented mapping, rather than branch oriented mapping (above)
BUILDING_NIGHT_REQUIREMENTS = {
    "Toren_west": {
        "branches": ["0", "1", "2"],
        "min_building": 1
    },
    "Toren_oost": {
        "branches": ["4", "5", "6"],
        "min_building": 1
    },
    "Zuid": {
        "branches": ["3"],
        "min_building": 0
    },
    "Aanleun": {
        "branches": ["7"],
        "min_building": 0
    }
}


from collections import defaultdict
BUILDING_TO_BRANCHES = defaultdict(set)
for branch, building in BRANCH_TO_BUILDING.items():
    BUILDING_TO_BRANCHES[building].add(branch)
    
print(SHIFT_TYPES.items())

# Dictionary to store min_staff per time slot per branch
TIME_SLOT_TO_MIN_STAFF = {str(b): {} for b in SHIFT_TYPES.keys()}

# Loop door elke branch
for store_id, shifts in SHIFT_TYPES.items():
    # Bepaal de tijdstippen waarvoor er überhaupt een shift bestaat in deze winkel
    active_time_slots = set()

    for shift in shifts:
        start_time = shift.start_time
        end_time = (start_time + shift.duration) % 24  # Houd rekening met overloop over middernacht

        if start_time <= end_time:
            # Normale shift
            active_time_slots.update(np.arange(start_time, end_time, 0.25))
        else:
            # Shift die over middernacht heen gaat
            active_time_slots.update(np.arange(start_time, 24, 0.25))
            active_time_slots.update(np.arange(0, end_time, 0.25))

    # Haal de juiste shiftperioden op voor deze winkel
    shift_periods = SHIFT_PERIODS_PER_BRANCH.get(store_id, [])

    # Vul de dictionary met de minimale staffing per tijdslot, alleen als er een shift actief is
    for time_slot in TIME_SLOTS:
        if time_slot in active_time_slots:
            # Standaardwaarde als er geen specifieke staffing vereist is
            min_staff = 0

            # Bepaal de minimale staffing per tijdslot uit de winkel-specifieke shiftperioden
            for period in shift_periods:
                start = period["start"]
                end = period["end"]
                required = period["min_staff"]

                # Controleer of het tijdslot binnen de shiftperiode valt
                if start <= end:
                    # Normale shift die niet over middernacht gaat
                    if start <= time_slot < end:
                        min_staff = required
                else:
                    # Nachtshift die over middernacht heen gaat
                    if time_slot >= start or time_slot < end:
                        min_staff = required

            TIME_SLOT_TO_MIN_STAFF[store_id][round(time_slot, 2)] = min_staff
        else:
            # Geen shift actief op dit tijdstip, dus vereist personeel is 0
            TIME_SLOT_TO_MIN_STAFF[store_id][round(time_slot, 2)] = 0

# Define function coverage requirement for particular functions (VPK)
#{'bbl helpende': 0, 'bbl verz ig': 1, 'bbl vpk': 2, 'helpende': 3, 'helpende plus': 4, 'verz ig': 5, 'vpk': 6, 'vpk in opl': 7, 'woonass': 8}
FUNCTION_COVERAGE_REQUIREMENTS = [
    {
        "branches": {0, 1, 2, 3, 4, 5, 6, 7},          # Lijst van afdelingen waarop de voorwaarde globaal moet worden voldaan
        "function_ids": [2, 6, 7],            # Vereiste functies
        "qualified_ids": [5,6],               # Functies die zelfstadig een verantwoorde dienst mogen draaien
        "min_staff": 2,                       # Minimaal aantal medewerkers met de betreffende functies die in de genoemde afdelingen aanwezig moeten zijn
        "time_window": (7.0, 23.0),           # Tijdslot waarin de voorwaarde geldt. 
    },
    {
        "branches": {0, 1, 2, 4, 5, 6},
        "function_ids": [0,1,2,3,4,5,6,7],    # Allowed functions
        "qualified_ids": [5,6],               # Functies die zelfstadig een verantwoorde dienst mogen draaien
        "min_staff": 2,
        "time_window": (23.75, 2.0),              # Night shift coverage
    },
]


def shift_overlaps(start1, end1, start2, end2):
    """Returns True if shift [start1, end1) overlaps window [start2, end2) on 24h clock."""
    start1 %= 24
    end1 %= 24
    start2 %= 24
    end2 %= 24

    if start1 < end1:
        range1 = lambda t: start1 <= t < end1
    else:
        range1 = lambda t: t >= start1 or t < end1

    return range1(start2) or range1(end2 - 0.01)

def shift_is_night_shift(shift_start, shift_end):
    return shift_start >= 22


MIN_REST_HOURS = 14   # Shifts typically span less than 9 hours, hence 24-9=15 > 14 TODO: check of meer dan 24 uur werkt

# Define the combinations of shifts that are incompatible as subsequent shifts
# Format: {branch: [(s_prev, s_next), ...]}
SHIFT_INCOMPATIBLE_PAIRS = {}

for branch, shifts in SHIFT_TYPES.items():
    incompatible = []
    for i, shift1 in enumerate(shifts):
        start1 = shift1.start_time
        end1   = (start1 + shift1.duration) % 24
        for j, shift2 in enumerate(shifts):
            start2 = shift2.start_time

            # ⏱ Calculate rest between end of shift1 and start of shift2 the next day
            rest_hours = 24 + (start2  - end1)  # Neem een dag tijd (dit is namelijk altijd het geval) en tel daar het verschil in nieuw begin en oude eindtijd bij op. 

            if rest_hours < MIN_REST_HOURS: #and not shift_is_night_shift(shift_start=start1):
                incompatible.append((i, j))
            # if shift_is_night_shift(shift_start=start1):
            #     NIGHT_TIME_REST = 
    SHIFT_INCOMPATIBLE_PAIRS[branch] = incompatible


for building_name, config in BUILDING_NIGHT_REQUIREMENTS.items():
    branches = config["branches"]
    min_staff = config["min_building"]


# ## Solver met alle hard en soft constraints. 
# ###### Constraints 4a en 8b zijn van hard naar zacht gezet

# In[ ]:


#%pip install ortools


# #### Specificeer inputs voor soft constraints

# In[29]:


MAX_CONSECUTIVE_DAYS                = 5     # Hoeveel dagen er achter elkaar gewerkt mag worden
overwork_penalty_factor             = 5 
spread_penalty_factor               = 40 
weekday_variance_penalty_factor     = 70 # TODO: in uren
shift_overload_penalty_factor       = 200000 # TODO: CHECK OF DEZE WEGEN MET 3 DAGEN UIT 5 GOED WERKT VOOR BLOKJES VAN 3
consecutive_workdays_penalty_factor = 30 # consecutive workdays adaptief maken o.b.v. uren: delen door 8 en afronden
total_hours_penalty_factor          = 1
blacklist_penalty_factor            = 75000 # aan
free_weekday_penalty_factor         = 1500000 # aan
free_nv_penalty_factor              = 4000 # uit
scattered_shift_penalty_factor      = 3000 
branch_distribution_penalty_factor  = 2
flex_worker_penalty_factor          = 10
underassignment_penalty_factor      = 9
meeting_distance_penalty_factor     = 20
weekend_cohesion_penalty_factor     = 30000
evv_day_shift_penalty_factor        = 12
non_evv_balance_penalty_factor      = 40000
consecutive_night_penalty_factor    = 1000000
consecutive_weekend_penalty_factor  = 5000
print(BUILDING_TO_BRANCHES.items())
print(TIME_SLOT_TO_MIN_STAFF.get(str(6), {}).get(round(1, 1), 0))


# In[30]:


from ortools.sat.python import cp_model
import datetime
from collections import defaultdict
import math

# no_shifts = len(SHIFT_TYPES)
# Scheduler klasse die OR-Tools gebruikt

from ortools.sat.python import cp_model

class DebugCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, works, employees, shift_types):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.works = works
        self.employees = employees
        self.shift_types = shift_types
        self.best_schedule = []

    def on_solution_callback(self):
        print(f"\n✅ Solution found at {self.WallTime():.2f}s")
        
        # Store a feasible solution
        schedule = []
        for (e, d, s, b), var in self.works.items():
            if self.Value(var):
                shift = self.shift_types.get(str(b), [])[s]
                employee = self.employees[e]
                shift_end = (shift.start_time + shift.duration) % 24
                print(f"  - {employee.name} (func: {employee.func}) → Day {d + 1}, Branch {b}, Shift: {shift.start_time:.2f}–{shift_end:.2f}")
                schedule.append((e, employee, d, s, b, shift.start_time, shift.duration))
        self.best_schedule = schedule

    def get_best_schedule(self):
        return self.best_schedule


class OrToolsScheduler:
    def __init__(self, employees, year, month):
        self.employees = employees
        # Sort employees in descending order by priority (higher priority first)
        self.employees.sort(key=lambda emp: emp.priority, reverse=True)
        self.num_employees = len(employees)
        self.num_days = get_days_in_month(year, month)
        self.all_branches = sorted({branch for emp in employees for branch in emp.allowed_branches})
        self.model = cp_model.CpModel()

        # ✅ Decision variables: works[e, d, s, b] is True if employee e works shift s at branch b on day d
        self.works = {}
        self.weekly_hours = {}

        for e in range(self.num_employees):
            for d in range(self.num_days):
                for b in self.employees[e].allowed_branches:  # ✅ Only assign shifts for valid branches
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):  # ✅ Get valid shifts per branch
                        var_name = f"work{e}_{d}_{s}_{b}"
                        self.works[(e, d, s, b)] = self.model.NewBoolVar(var_name)

        # ✅ Weekly hours tracking per branch
        for e in range(self.num_employees):
            for w in range(WEEKS_PER_MONTH):
                self.weekly_hours[(e, w)] = self.model.NewIntVar(
                    0, int(self.employees[e].contract_hours),
                    f"weekly_hours_{e}_{w}"
                )

        # ✅ Add constraints based on the updated structure
        self._add_constraints()

               
   

    def _add_constraints(self):
        unique_branches = {branch for emp in self.employees for branch in emp.allowed_branches}
        print(unique_branches)
        # Define key time slots (you can adjust granularity here)
        KEY_TIME_SLOTS = np.arange(7, 24, 1)  # Every 2 hours between 06:00 and 22:00

        # 1a) Minimal coverage requirement per branch, day, and key time slot
        for branch in unique_branches:
            for d in range(self.num_days):
                for t in KEY_TIME_SLOTS:
                    min_staff = TIME_SLOT_TO_MIN_STAFF.get(str(branch), {}).get(round(t, 1), 0)
                    if min_staff == 0:
                        continue

                    staff_covering_slot = []
                    for e in range(self.num_employees):
                        # Initialize function id for later use inside of shift loop
                        employee_func_id = self.employees[e].func
                        
                        
                        if branch not in self.employees[e].allowed_branches:
                            continue

                        for s, shift in enumerate(SHIFT_TYPES.get(str(branch), [])):
                            # Make sure only allowed functions are assigned to shifts
                            if employee_func_id not in shift.allowed_functions:            # THIS LINE MAKES THE SOLUTION INFEASIBLE
                                continue
                            
                            # Initialize shift time values
                            shift_start = shift.start_time
                            shift_end = (shift.start_time + shift.duration) % 24

                            # ✅ Only include non-night shifts
                            if shift_start >= shift_end:
                                continue  # Night shift: skip!
                            
                            if shift_start <= t < shift_end and (e, d, s, branch) in self.works:
                                staff_covering_slot.append(self.works[(e, d, s, branch)])
                                #print("day ASSIGNED")
                    if staff_covering_slot:
                        #print(f"Medewerker: ", {e}, " dag: ", {d + 1}, " shift: ", {s}, " branch: ", {branch})
                        self.model.Add(sum(staff_covering_slot) >= min_staff)
                        
        # 1b) Nacht shifts worden apart opgenomen om per vleugel de voorwaarden te verwerken.                               
        for building_name, config in BUILDING_NIGHT_REQUIREMENTS.items():
            branches = config["branches"]
            min_staff = config["min_building"]

            for d in range(self.num_days):
                employee_night_vars = []

                for e, employee in enumerate(self.employees):
                    if not employee.works_night_shifts:
                        continue  # Skip if employee doesn't work nights

                    # Track all night shift assignments for this employee on this day in this building
                    emp_shift_vars = []

                    for b in branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            shift_start = shift.start_time
                            shift_end = (shift.start_time + shift.duration) % 24

                            # Only consider true night shifts
                            if shift_start >= 23 or shift_end < shift_start:
                                key = (e, d, s, int(b))
                                if key in self.works:
                                    emp_shift_vars.append(self.works[key])

                    if emp_shift_vars:
                        # One shift max per employee across all branches in this building
                        emp_night_assigned = self.model.NewBoolVar(f"assigned_night_e{e}_d{d}_bldg_{building_name}")
                        self.model.AddMaxEquality(emp_night_assigned, emp_shift_vars)
                        employee_night_vars.append(emp_night_assigned)

                # Enforce exactly `min_staff` employees assigned in this building on this night
                if employee_night_vars:
                    self.model.Add(sum(employee_night_vars) == min_staff)


        # 1c) Voeg CAO gebonden limieten toe aan opeenvolgende nachtshifts: na 3 nachtshifts if 46 (2d) rust verplicht.  
        for e, employee in enumerate(self.employees):
            if not employee.works_night_shifts:
                continue  # Skip employees who can't work nights

            for d in range(self.num_days - 3):  # Leave room for streak + 2 rest days
                # --- Gather night shifts over a 4-day sliding window ---
                night_vars_4day = []
                for offset in range(4):
                    day = d + offset
                    for b in employee.allowed_branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            start = shift.start_time
                            end = (shift.start_time + shift.duration) % 24
                            if start >= 23 or end < start:
                                key = (e, day, s, int(b))
                                if key in self.works:
                                    night_vars_4day.append(self.works[key])

                # --- Hard constraint: never allow more than 3 night shifts in any 4-day window ---
                if night_vars_4day:
                    self.model.Add(sum(night_vars_4day) <= 3)

                # --- Detect 3-night streaks ---
                night_vars_3day = []
                for offset in range(3):
                    day = d + offset
                    for b in employee.allowed_branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            start = shift.start_time
                            end = (shift.start_time + shift.duration) % 24
                            if start >= 23 or end < start:
                                key = (e, day, s, int(b))
                                if key in self.works:
                                    night_vars_3day.append(self.works[key])

                trigger_3 = self.model.NewBoolVar(f"night_3_streak_e{e}_d{d}")
                self.model.Add(sum(night_vars_3day) == 3).OnlyEnforceIf(trigger_3)
                self.model.Add(sum(night_vars_3day) != 3).OnlyEnforceIf(trigger_3.Not())

                # --- Detect 2-night streaks ---
                night_vars_2day = []
                for offset in range(2):
                    day = d + offset
                    for b in employee.allowed_branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            start = shift.start_time
                            end = (shift.start_time + shift.duration) % 24
                            if start >= 23 or end < start:
                                key = (e, day, s, int(b))
                                if key in self.works:
                                    night_vars_2day.append(self.works[key])

                # Make sure it's exactly 2 nights AND not part of a 3-night streak
                next_night = []
                day_after_2 = d + 2
                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        start = shift.start_time
                        end = (shift.start_time + shift.duration) % 24
                        if start >= 23 or end < start:
                            key = (e, day_after_2, s, int(b))
                            if key in self.works:
                                next_night.append(self.works[key])

                trigger_2 = self.model.NewBoolVar(f"night_2_streak_e{e}_d{d}")
                self.model.Add(sum(night_vars_2day) == 2).OnlyEnforceIf(trigger_2)
                if next_night:
                    self.model.Add(sum(next_night) == 0).OnlyEnforceIf(trigger_2)

                # Create helper BoolVars to represent the logic
                exactly_2_nights = self.model.NewBoolVar(f"exactly_2_nights_e{e}_d{d}")
                next_night_exists = self.model.NewBoolVar(f"next_night_exists_e{e}_d{d}")

                # Link to actual values
                self.model.Add(sum(night_vars_2day) == 2).OnlyEnforceIf(exactly_2_nights)
                self.model.Add(sum(night_vars_2day) != 2).OnlyEnforceIf(exactly_2_nights.Not())

                if next_night:
                    self.model.Add(sum(next_night) > 0).OnlyEnforceIf(next_night_exists)
                    self.model.Add(sum(next_night) == 0).OnlyEnforceIf(next_night_exists.Not())
                else:
                    # No third night possible — so no overlap
                    self.model.Add(next_night_exists == 0)

                # Now use the boolean variables in the OR
                self.model.AddBoolOr([
                    exactly_2_nights.Not(),
                    next_night_exists
                ]).OnlyEnforceIf(trigger_2.Not())


                # --- Enforce rest on day d+3 and d+4 if 3-night streak ---
                for rest_offset in [3, 4]:
                    rest_day = d + rest_offset
                    if rest_day >= self.num_days:
                        continue

                    rest_vars = [
                        self.works[(e, rest_day, s, b)]
                        for b in employee.allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, rest_day, s, b) in self.works
                    ]
                    if rest_vars:
                        self.model.Add(sum(rest_vars) == 0).OnlyEnforceIf(trigger_3)

                # --- Enforce rest on day d+2 and d+3 if 2-night streak ---
                for rest_offset in [2, 3]:
                    rest_day = d + rest_offset
                    if rest_day >= self.num_days:
                        continue

                    rest_vars = [
                        self.works[(e, rest_day, s, b)]
                        for b in employee.allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, rest_day, s, b) in self.works
                    ]
                    if rest_vars:
                        self.model.Add(sum(rest_vars) == 0).OnlyEnforceIf(trigger_2)
                        
            # === Detect isolated night shifts and enforce 2 rest days ===
            for d in range(1, self.num_days - 2):  # Need d-1, d, d+1, d+2
                prev_night = []
                curr_night = []
                next_night = []

                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        start = shift.start_time
                        end = (start + shift.duration) % 24
                        if start >= 23 or end < start:
                            if (e, d - 1, s, int(b)) in self.works:
                                prev_night.append(self.works[(e, d - 1, s, int(b))])
                            if (e, d, s, int(b)) in self.works:
                                curr_night.append(self.works[(e, d, s, int(b))])
                            if (e, d + 1, s, int(b)) in self.works:
                                next_night.append(self.works[(e, d + 1, s, int(b))])

                # Define BoolVars to represent night activity
                has_prev = self.model.NewBoolVar(f"has_prev_night_e{e}_d{d}")
                has_curr = self.model.NewBoolVar(f"has_curr_night_e{e}_d{d}")
                has_next = self.model.NewBoolVar(f"has_next_night_e{e}_d{d}")

                if prev_night:
                    self.model.AddMaxEquality(has_prev, prev_night)
                else:
                    self.model.Add(has_prev == 0)

                if curr_night:
                    self.model.AddMaxEquality(has_curr, curr_night)
                else:
                    self.model.Add(has_curr == 0)

                if next_night:
                    self.model.AddMaxEquality(has_next, next_night)
                else:
                    self.model.Add(has_next == 0)

                # Trigger only if isolated: [0, 1, 0]
                pattern_010  = self.model.NewBoolVar(f"isolated_night_e{e}_d{d}")
                self.model.AddBoolAnd([
                    has_prev.Not(),
                    has_curr,
                    has_next.Not()
                ]).OnlyEnforceIf(pattern_010)
                self.model.AddBoolOr([
                    has_prev,
                    has_curr.Not(),
                    has_next
                ]).OnlyEnforceIf(pattern_010 .Not())
                
                # Trigger if [1, 0, 1] (sandwiched rest)
                pattern_101 = self.model.NewBoolVar(f"pattern_101_e{e}_d{d}")
                self.model.AddBoolAnd([
                    has_prev,
                    has_curr.Not(),
                    has_next
                ]).OnlyEnforceIf(pattern_101)
                self.model.AddBoolOr([
                    has_prev.Not(),
                    has_curr,
                    has_next.Not()
                ]).OnlyEnforceIf(pattern_101.Not())

                # Combined trigger: either [0, 1, 0] or [1, 0, 1]
                trigger = self.model.NewBoolVar(f"undesirable_night_pattern_e{e}_d{d}")
                self.model.AddBoolOr([pattern_010, pattern_101]).OnlyEnforceIf(trigger)
                self.model.AddBoolAnd([pattern_010.Not(), pattern_101.Not()]).OnlyEnforceIf(trigger.Not())
                
                # Enforce 2 rest days after isolated night
                for rest_offset in [1, 2]:
                    rest_day = d + rest_offset
                    if rest_day >= self.num_days:
                        continue
                    rest_vars = [
                        self.works[(e, rest_day, s, b)]
                        for b in employee.allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, rest_day, s, b) in self.works
                    ]
                    if rest_vars:
                        self.model.Add(sum(rest_vars) == 0).OnlyEnforceIf(trigger)            

        # 2) Max 1 shift per dag per medewerker: som van alle shifts op dag d voor medewerker e <= 1. Tel nachtshifts op beide dagen mee om te voorkomen dat mensen te veel shifts krijgen
        for e in range(self.num_employees):
            for d in range(self.num_days):
                shift_vars = []

                # All shifts on day d
                for b in self.employees[e].allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        key = (e, d, s, b)
                        if key in self.works:
                            shift_vars.append(self.works[key])

                # Block early-start shifts after a night shift the day before
                if d > 0:
                    for b in self.employees[e].allowed_branches:
                        for s_prev, prev_shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            if prev_shift.start_time >= 22 or (prev_shift.start_time + prev_shift.duration) % 24 < prev_shift.start_time:
                                prev_key = (e, d - 1, s_prev, b)
                                if prev_key in self.works:
                                    # For each potentially conflicting shift on d:
                                    for s_curr, curr_shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                                        # Block next-day shift if it starts before 14:00 (or any cutoff you prefer)
                                        if curr_shift.start_time < 14:
                                            curr_key = (e, d, s_curr, b)
                                            if curr_key in self.works:
                                                # Night shift yesterday + early shift today → not allowed
                                                self.model.AddBoolOr([
                                                    self.works[prev_key].Not(),
                                                    self.works[curr_key].Not()
                                                ])
                # Still enforce: max 1 shift per day
                self.model.Add(sum(shift_vars) <= 1)

                # # Also prevent shift the day before from affecting current day due to night shift overlap
                # if d > 0:
                #     for b in self.employees[e].allowed_branches:
                #         for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                #             if shift.start_time >= 22 or shift.start_time + shift.duration > 24:
                #                 if (e, d - 1, s, b) in self.works:
                #                     shift_sum += self.works[(e, d - 1, s, b)]

                # self.model.Add(shift_sum <= 1)


                                              
        # 3a) Check of het aantal gewerkte uren binnen het contract niet wordt overschreden door som van geplande uren (shift gebaseerd)
        for e in range(self.num_employees):
            max_hours = int(self.employees[e].net_contract_hours * max_hour_limit)  # Adjust contract hours with limit
        
            # ✅ Define 'b' before using it inside SHIFT_TYPES.get(b, [])
            self.model.Add(
                sum(
                    self.works[(e, d, s, b)] * int(shift.duration)  # ✅ Use correct shift duration reference
                    for b in self.employees[e].allowed_branches  # ✅ Define 'b' first
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), []))  # ✅ Use 'b' after it's defined
                    for d in range(self.num_days)  # ✅ Iterate over days after shift details are set
                   if (e, d, s, b) in self.works  # ✅ Ensure valid index
                ) <= max_hours
            )

        # 3b) Limiteer het aantal shifts dat iemand achter elkaar wordt ingeroosterd. 
        for e in range(self.num_employees):
            max_consecutive_days = math.ceil(self.employees[e].contract_hours / 4.3 / 8)
            for start_day in range(self.num_days - max_consecutive_days):
                worked_days = []
                
                for offset in range(max_consecutive_days + 1):  # Check 6-day windows
                    day = start_day + offset
                    works_any_shift = self.model.NewBoolVar(f"works_e{e}_d{day}")
                    
                    self.model.AddMaxEquality(
                        works_any_shift,
                        [
                            self.works[(e, day, s, b)]
                            for b in self.employees[e].allowed_branches
                            for s in range(len(SHIFT_TYPES.get(str(b), [])))
                            if (e, day, s, b) in self.works
                        ]
                    )
                    worked_days.append(works_any_shift)

                # Prevent working 6 consecutive days
                self.model.Add(sum(worked_days) <= max_consecutive_days)


        # 4) Hou opgegeven vrije dagen altijd vrij 
        # 4a1) blokkeer de vaste vrije dagen (e.g. 'elke vrijdag vrij') # INFEASIBLE
        # for e, employee in enumerate(self.employees):
        #   for d in range(self.num_days):
        #        # 🔹 Get actual weekday (Monday = 0, Sunday = 6)
        #        day_of_week = datetime.date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).isoweekday()  # Convert to (Monday=1, Sunday=7)#
        #        #print(e, employee, d, day_of_week, unavailable_days)
        #        # 🔹 If the employee has this day off, prevent shift assignment
        #        if day_of_week in employee.unavailable_days:
                   
        #            for b in employee.allowed_branches:  # ✅ Ensure this applies across branches
        #               for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])): 
        #                   self.model.Add(self.works[(e, d, s, b)] == 0)
        #        else: 
        #            continue
        
        MAX_UNAVAILABLE_DAY_ASSIGNMENTS = 1  # Or customize per employee

        for e, employee in enumerate(self.employees):
            unavailable_violations = []

            for d in range(self.num_days):
                weekday = datetime.date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).isoweekday()  # 1–7 (Mon–Sun)

                if weekday in employee.unavailable_days:
                    for b in employee.allowed_branches:
                        for s in range(len(SHIFT_TYPES.get(str(b), []))):
                            key = (e, d, s, b)
                            if key in self.works:
                                unavailable_violations.append(self.works[key])

            if unavailable_violations:
                self.model.Add(sum(unavailable_violations) <= MAX_UNAVAILABLE_DAY_ASSIGNMENTS)

               
        # 4a2) blokkeer oneven wensen
        # for e, employee in enumerate(self.employees):
        #     for d in employee.blacklisted_days:
        #         for b in employee.allowed_branches:  # ✅ Ensure this applies across branches
        #             for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])): 
        #                 self.model.Add(self.works[(e, int(d)-1, s, b)] == 0)
    
        
        # 4b) Blokkeer de scholingsuren
        for e, employee in enumerate(self.employees):

            restrictions = employee.blocked_hours  # Convert string to dict
            #print(restrictions)
            
            for restriction in restrictions:
                #print(restriction)
                for day_str, time_range in restriction.items():
                    day = int(day_str) - 1
                    if 0 <= day < self.num_days:
                        try:
                            start_str, end_str = time_range.split("-")
                            block_start = float(start_str)
                            block_end = float(end_str)
                        except ValueError:
                            continue  # skip malformed entries

                        for b in employee.allowed_branches:
                            for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                                shift_start = shift.start_time
                                shift_end = (shift.start_time + shift.duration) % 24

                                # Check overlap between [block_start, block_end] and shift
                                overlap = (
                                    (shift_start < block_end and shift_end > block_start)
                                    if shift_start < shift_end
                                    else (shift_start < block_end or shift_end > block_start)
                                )

                                if overlap and (e, day, s, b) in self.works:
                                    #print(f"Medewerker: ", {e}, " dag: ", {day + 1}, " shift: ", {s}, " branch: ", {b})
                                    self.model.Add(self.works[(e, day, s, b)] == 0)
                                        
        # 4c) Zorg ervoor dat medewerkers alleen werken in de opgegeven tijden:
        for e, employee in enumerate(self.employees):
            preferences = employee.preferred_times  # Format: [{"1": "7-13"}, {"3": "9-17"}]
            if not preferences:
                continue  # Skip if no preference given
            
            # Merge multiple time windows for each weekday into a list
            preferred_time_by_weekday = {}

            for pref in preferences:
                for k, v in pref.items():
                    k = k.strip()
                    if k not in preferred_time_by_weekday:
                        preferred_time_by_weekday[k] = []
                    preferred_time_by_weekday[k].append(v.strip())
        
            for d in range(self.num_days):
                weekday = str(get_weekday(REFERENCE_YEAR, REFERENCE_MONTH, d))  # 1 = Monday

                if weekday not in preferred_time_by_weekday:
                    continue

                for b in self.employees[e].allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        shift_start = shift.start_time
                        shift_end = (shift.start_time + shift.duration) % 24

                        # If it overlaps with at least one preferred window, it's okay
                        allowed = False
                        for time_range in preferred_time_by_weekday[weekday]:
                            pref_start, pref_end = map(float, time_range.split("-"))
                            if shifts_in_preferred_time(shift_start, shift_end, pref_start, pref_end):
                                # if employee.name == 'marijke postma':
                                #     print(shift_start, shift_end, pref_start, pref_end)
                                allowed = True
                                break  # at least one good overlap → assignable

                        is_night = shift_start >= 22 or shift_end < shift_start  # Night shifts cross midnight

                        # ❌ Block if not allowed by time preference, unless it's a night shift and the employee accepts night shifts
                        if not allowed and (not is_night or not employee.works_night_shifts):
                            if (e, d, s, b) in self.works:
                                self.model.Add(self.works[(e, d, s, b)] == 0)
                                      
                         
        # 5a) Zorg ervoor dat verantwoorde shifts altijd gevuld worden.
        for d in range(self.num_days):
            for b in self.all_branches:
                for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                    if shift.presence_mandatory:
                        self.model.Add(
                            sum(
                                self.works[(e, d, s, b)]
                                for e in range(self.num_employees)
                                if b in self.employees[e].allowed_branches
                                and self.employees[e].func in shift.allowed_functions
                                and (e, d, s, b) in self.works
                            ) >= 1
                        )
                        
        # 5b) Check of er wel een feasible rooster is onder de beschikbaarheid voor de verantwoorde shifts
        eligible_employees = [
            self.works[(e, d, s, b)]
            for e in range(self.num_employees)
            if b in self.employees[e].allowed_branches
            and self.employees[e].func in shift.allowed_functions
            and (e, d, s, b) in self.works
        ]

        if shift.presence_mandatory and eligible_employees:
            self.model.Add(sum(eligible_employees) >= 1)
        elif shift.presence_mandatory:
            print(f"⚠️ No eligible employee found for mandatory shift on day {d}, branch {b}, shift {s}")
            
        # 5c) Pas de regels toe voor elke functie (e.g. altijd een vpk'er in huis, mensen in opleiding niet alleen etc.)
        for rule in FUNCTION_COVERAGE_REQUIREMENTS:
            covered_branches = rule["branches"]
            func_ids = rule["function_ids"]
            qualified_ids = rule["qualified_ids"]
            min_staff = rule["min_staff"]
            start_t, end_t = rule["time_window"]

            is_night_rule = start_t > end_t

            for d in range(self.num_days):
                total_staff = []
                qualified_staff = []

                for b in covered_branches:
              
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        shift_start = shift.start_time
                        shift_end = (shift.start_time + shift.duration) % 24

                        if shift_overlaps(shift_start, shift_end, start_t, end_t):
                            # Determine correct "effective" day this shift contributes coverage to
                            if is_night_rule and shift_start < end_t:
                                effective_day = d + 1  # Early morning portion counts for next day
                            else:
                                effective_day = d

                            if effective_day >= self.num_days:
                                continue  # Skip if out of range

                            for e, emp in enumerate(self.employees):
                                branch_allowed = (b in emp.allowed_branches) if not is_night_rule else True

                                if (
                                    branch_allowed
                                    and emp.func in func_ids
                                    and (e, d, s, b) in self.works
                                ):
                                    var = self.works[(e, d, s, b)]

                                    if effective_day == d:
                                        total_staff.append(var)
                                        if emp.func in qualified_ids:
                                            qualified_staff.append(var)
                                    else:
                                        # Make new lists if needed for d+1
                                        # (assuming you're inside a per-day loop)
                                        # If you want global tracking: use a dict per day
                                        total_staff.append(var)
                                        if emp.func in qualified_ids:
                                            qualified_staff.append(var)

                # Enforce: at least min_staff working, and at least one qualified person
                self.model.Add(sum(total_staff) >= min_staff)
                self.model.Add(sum(qualified_staff) >= 1) 
         
                
        # for rule in FUNCTION_COVERAGE_REQUIREMENTS:
        #     covered_branches = rule["branches"]
        #     func_ids = rule["function_ids"]
        #     qualified_ids = rule["qualified_ids"]
        #     min_staff = rule["min_staff"]
        #     start_t, end_t = rule["time_window"]

        #     is_night_rule = start_t > end_t  # used to control branch filtering

        #     for d in range(self.num_days):
        #         total_staff = []
        #         qualified_staff = []

        #         for b in covered_branches:

        #             for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
        #                 shift_start = shift.start_time
        #                 shift_end = (shift.start_time + shift.duration) % 24

        #                 if shift_overlaps(shift_start, shift_end, start_t, end_t):
        #                     for e, emp in enumerate(self.employees):
        #                         # Branch restriction applies only if not a night rule
        #                         branch_allowed = (b in emp.allowed_branches) if not is_night_rule else True

        #                         if (
        #                             branch_allowed
        #                             and emp.func in func_ids
        #                             and (e, d, s, b) in self.works
        #                         ):
        #                             var = self.works[(e, d, s, b)]
        #                             total_staff.append(var)

        #                             if emp.func in qualified_ids:
        #                                 qualified_staff.append(var)

        #         # Enforce constraints per rule and day
        #         self.model.Add(sum(total_staff) >= min_staff)
        #         self.model.Add(sum(qualified_staff) >= 1)
                
        
        # for rule in FUNCTION_COVERAGE_REQUIREMENTS:
        #     covered_branches = rule["branches"]
        #     func_ids = rule["function_ids"]
        #     qualified_ids = rule["qualified_ids"]
        #     min_staff = rule["min_staff"]
        #     start_t, end_t = rule["time_window"]

        #     for d in range(self.num_days):
        #         total_staff = []
        #         qualified_staff = []

        #         for b in covered_branches:
        #             for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
        #                 # Check if shift overlaps with time window
        #                 shift_start = shift.start_time
        #                 shift_end = (shift.start_time + shift.duration) % 24

        #                 if start_t < end_t:
        #                     covers_time = (shift_start < end_t) and (shift_end > start_t)
        #                     day = True
        #                 else: 
        #                     covers_time = (shift_start < end_t) or (shift_end < start_t) # nacht shift begint voor 24:00, dus shift end is altijd kleiner dan start_t = 23.75
        #                     day = False
                            
        #                 if covers_time and day:
        #                     for e in range(self.num_employees):
        #                         emp = self.employees[e]

        #                         if ( # Consider day time shifts
        #                             b in emp.allowed_branches
        #                             and emp.func in func_ids
        #                             and (e, d, s, b) in self.works
        #                         ):
        #                             total_staff.append(self.works[(e, d, s, b)])
        #                             #print('total_staff appended')

        #                             if emp.func in qualified_ids:
        #                                 qualified_staff.append(self.works[(e, d, s, b)])
        #                                 #print('qualified_staff appended')

        #                 if covers_time and not day: # covers night time: # Make this condition more specific
        #                     for e in range(self.num_employees):
        #                         emp = self.employees[e]
 
        #                         if ( 
        #                             emp.func in func_ids
        #                             and (e, d, s, b) in self.works
        #                         ):
        #                             total_staff.append(self.works[(e, d, s, b)])

        #                             if emp.func in qualified_ids:
        #                                 qualified_staff.append(self.works[(e, d, s, b)])
                        
        #         # if total_staff and qualified_staff: # Deze regel is niet generiek genoeg. Later kan een functie nodig zijn om te bepalen of het moment een gekwalificeerd persoon heeft
        #         self.model.Add(sum(total_staff) >= min_staff)
        #         self.model.Add(sum(qualified_staff) >= 1)  # Must have at least one fully qualified

                    
        # 5d) Forceer dat alle functie van medewerker goed worden meegenomen
        for e, employee in enumerate(self.employees):
            emp_func = employee.func
            for d in range(self.num_days):
                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        key = (e, d, s, b)
                        if key in self.works:
                            if emp_func not in shift.allowed_functions:
                                # ❌ Not allowed to work this shift with this function
                                self.model.Add(self.works[key] == 0)
                                
                             # ❌ Rule 2: Block night shifts if employee doesn't work nights
                            shift_start = shift.start_time
                            shift_end = (shift.start_time + shift.duration) % 24
                            is_night_shift = shift_start >= 23 or shift_end < shift_start

                            if is_night_shift and not employee.works_night_shifts:
                                self.model.Add(self.works[key] == 0)

            
        

        # 6) Zorg ervoor dat de opeenvolgende shifts genoeg tijdsverschil hebben om late diensten en nachtdiensten niet op te laten volgen door vroege diensten     
        for e in range(self.num_employees):
            for d in range(self.num_days - 1):
                # Loop through all branches and shift types for day d (first shift)
                for b1 in self.employees[e].allowed_branches:
                    for s1, shift1 in enumerate(SHIFT_TYPES.get(str(b1), [])):
                        start1 = shift1.start_time
                        end1 = (shift1.start_time + shift1.duration) % 24

                        for b2 in self.employees[e].allowed_branches:
                            for s2, shift2 in enumerate(SHIFT_TYPES.get(str(b2), [])):
                                start2 = shift2.start_time
                                # ✅ Calculate hours between shifts
                                hours_between = (start2 - end1) % 24
                                
                                # 🚫 Too short turnaround (e.g. less than 14 hours between shifts)
                                if hours_between < MIN_REST_HOURS:
                                    if (e, d, s1, b1) in self.works and (e, d+1, s2, b2) in self.works:
                                        self.model.Add(
                                            self.works[(e, d, s1, b1)] + self.works[(e, d+1, s2, b2)] <= 1
                                        )

        
        for e in range(self.num_employees):
           for d in range(self.num_days - 1):
               for b in self.employees[e].allowed_branches:
                   for s1, s2 in SHIFT_INCOMPATIBLE_PAIRS.get(str(b), []):
                       if (e, d, s1, b) in self.works and (e, d+1, s2, b) in self.works:
                           self.model.Add(self.works[(e, d, s1, b)] + self.works[(e, d + 1, s2, b)] <= 1) # <= 1 betekent dat maar een van de shifts (s1, s2) gebruikt mogen worden                     
     
        
        # # 8a) Blokkeer shifts op vakantiedagen
        for e, employee in enumerate(self.employees):
            for d in employee.vacation: # Loop through the vacation days (1-31)
                for b in employee.allowed_branches:  # ✅ Ensure this applies across branches
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        self.model.Add(self.works[(e, int(d) - 1, s, b)] == 0)  # ✅ Prevent scheduling at any branch  # d-1 because index starts at 0
        # # 8b) Blokkeer shifts op vrije dagen niet verlof
        for e, employee in enumerate(self.employees):
            for d in employee.free_days: 
                if employee.contract_type.lower() != "vast":
                    for b in employee.allowed_branches:  # ✅ Ensure this applies across branches
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            self.model.Add(self.works[(e, int(d) - 1, s, b)] == 0)  # ✅ Prevent scheduling at any branch  # d-1 because index starts at 0
        
        # 9) limiteer het aantal shifts dat in een window kan worden toegewezen o.b.v. contracturen
        ROLLING_WINDOW = 7
        for e, employee in enumerate(self.employees):

            # Estimate maximum shifts per week
            estimated_max_weekly_shifts = 6 # math.ceil(employee.contract_hours / 4.3 / 8) + 2 # e.g. 20 hours -> 3 + 1 = 4 days per 7 days at most

            for start_day in range(self.num_days - ROLLING_WINDOW + 1):
                shift_assignments = []

                for d in range(start_day, start_day + ROLLING_WINDOW):
                    for b in employee.allowed_branches:
                        for s in range(len(SHIFT_TYPES.get(str(b), []))):
                            key = (e, d, s, b)
                            if key in self.works:
                                shift_assignments.append(self.works[key])

                if shift_assignments:
                    self.model.Add(sum(shift_assignments) <= estimated_max_weekly_shifts)

        # 10) Introduceer altijd een tweede rustdag nadat er een rustdag is opgetreden na een reeks van 3 of meer werkdagen.
        # for e, employee in enumerate(self.employees):
        #     for d in range(self.num_days - 1):  # Leave room for d+1
        #         # Day d (rest check)
        #         day_work_vars = [
        #             self.works[(e, d, s, b)]
        #             for b in employee.allowed_branches
        #             for s in range(len(SHIFT_TYPES.get(str(b), [])))
        #             if (e, d, s, b) in self.works
        #         ]

        #         if day_work_vars:
        #             is_rest_day = self.model.NewBoolVar(f"is_rest_day_e{e}_d{d}")
        #             self.model.AddMaxEquality(is_rest_day, day_work_vars)
        #             self.model.Add(is_rest_day == 0)
        #         else:
        #             is_rest_day = self.model.NewConstant(1)

        #         # Day d+1 must be rest if d is rest
        #         next_day_work_vars = [
        #             self.works[(e, d + 1, s, b)]
        #             for b in employee.allowed_branches
        #             for s in range(len(SHIFT_TYPES.get(str(b), [])))
        #             if (e, d + 1, s, b) in self.works
        #         ]

        #         if next_day_work_vars:
        #             for var in next_day_work_vars:
        #                 self.model.Add(var == 0).OnlyEnforceIf(is_rest_day)
        
        
        # Oude logica om mensen na 2 dagen twee rustdagen te geven
        # for e, employee in enumerate(self.employees):
        #     for d in range(2, self.num_days - 1):  # leave room for d-2 to d+1
        #         work_vars = []

        #         for offset in [-2, -1, 0, 1]:  # d-2, d-1, d, d+1
        #             day = d + offset
        #             day_work = []

        #             for b in employee.allowed_branches:
        #                 for s in range(len(SHIFT_TYPES.get(str(b), []))):
        #                     key = (e, day, s, b)
        #                     if key in self.works:
        #                         day_work.append(self.works[key])

        #             if day_work:
        #                 var = self.model.NewBoolVar(f"worked_e{e}_d{day}")
        #                 self.model.AddMaxEquality(var, day_work)
        #             else:
        #                 var = self.model.NewConstant(0)

        #             work_vars.append(var)

        #         worked_prev2 = work_vars[0]
        #         worked_prev1 = work_vars[1]
        #         is_rest_day = work_vars[2].Not()
        #         next_day = work_vars[3]

        #         # Trigger if pattern [1,1,0] (at least 2 consecutive workdays followed by rest)
        #         trigger = self.model.NewBoolVar(f"needs_2nd_rest_e{e}_d{d}")
        #         self.model.AddBoolAnd([
        #             worked_prev2,
        #             worked_prev1,
        #             is_rest_day
        #         ]).OnlyEnforceIf(trigger)

        #         self.model.AddBoolOr([
        #             worked_prev2.Not(),
        #             worked_prev1.Not(),
        #             work_vars[2]  # not a rest day
        #         ]).OnlyEnforceIf(trigger.Not())

        #         # Enforce second rest day
        #         self.model.Add(next_day == 0).OnlyEnforceIf(trigger)

        # Oude logica om mensen na 3 dagen twee rustdagen te geven
        for e, employee in enumerate(self.employees):
            for d in range(3, self.num_days - 1):  # leave room for d-3 to d+1
                work_vars = []

                for offset in [-3, -2, -1, 0, 1]:  # days d-3 to d+1
                    day = d + offset
                    day_work = []

                    for b in employee.allowed_branches:
                        for s in range(len(SHIFT_TYPES.get(str(b), []))):
                            key = (e, day, s, b)
                            if key in self.works:
                                day_work.append(self.works[key])

                    if day_work:
                        var = self.model.NewBoolVar(f"worked_e{e}_d{day}")
                        self.model.AddMaxEquality(var, day_work)
                    else:
                        var = self.model.NewConstant(0)

                    work_vars.append(var)  # order: [d-3, d-2, d-1, d, d+1]

                worked_prev3 = work_vars[0]
                worked_prev2 = work_vars[1]
                worked_prev1 = work_vars[2]
                is_rest_day = work_vars[3].Not()
                next_day = work_vars[4]

                # Trigger if [1,1,1,0] pattern occurs (3 workdays followed by a rest day)
                trigger = self.model.NewBoolVar(f"needs_2nd_rest_e{e}_d{d}")
                self.model.AddBoolAnd([
                    worked_prev3,
                    worked_prev2,
                    worked_prev1,
                    is_rest_day
                ]).OnlyEnforceIf(trigger)

                self.model.AddBoolOr([
                    worked_prev3.Not(),
                    worked_prev2.Not(),
                    worked_prev1.Not(),
                    work_vars[3]  # not a rest day
                ]).OnlyEnforceIf(trigger.Not())

                # Enforce: day d+1 must also be a rest day if trigger is active
                self.model.Add(next_day == 0).OnlyEnforceIf(trigger)
                
        # Logica om weekenden om de week af te wisselen:
        # WEEKEND_DAYS = {5, 6}

        # for e, employee in enumerate(self.employees):
        #     weekend_work_flags = []

        #     for w in range(WEEKS_PER_MONTH):
        #         weekend_shift_vars = []

        #         for d in range(w * 7, min((w + 1) * 7, self.num_days)):
        #             if date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).weekday() in WEEKEND_DAYS:
        #                 for b in employee.allowed_branches:
        #                     for s in range(len(SHIFT_TYPES.get(str(b), []))):
        #                         key = (e, d, s, b)
        #                         if key in self.works:
        #                             weekend_shift_vars.append(self.works[key])

        #         if weekend_shift_vars:
        #             worked_weekend = self.model.NewBoolVar(f"worked_weekend_e{e}_w{w}")
        #             self.model.AddMaxEquality(worked_weekend, weekend_shift_vars)
        #             weekend_work_flags.append(worked_weekend)
        #         else:
        #             weekend_work_flags.append(self.model.NewConstant(0))

        #     # Enforce: no consecutive weekends worked
        #     for w in range(len(weekend_work_flags) - 1):
        #         worked_this = weekend_work_flags[w]
        #         worked_next = weekend_work_flags[w + 1]

        #         both_weekends = self.model.NewBoolVar(f"consecutive_weekends_e{e}_w{w}")
        #         self.model.AddBoolAnd([worked_this, worked_next]).OnlyEnforceIf(both_weekends)
        #         self.model.AddBoolOr([worked_this.Not(), worked_next.Not()]).OnlyEnforceIf(both_weekends.Not())

        #         # ❗️Hard constraint: consecutive weekends not allowed
        #         self.model.Add(both_weekends == 0)

        # Hard constraint for enforcing weekend 'blokjes'
        # WEEKEND_DAYS = {5, 6}  # Saturday = 5, Sunday = 6

        # for e, employee in enumerate(self.employees):
        #     for w in range(WEEKS_PER_MONTH):
        #         sat_day = None
        #         sun_day = None

        #         # Find actual day indices for Saturday and Sunday in this week
        #         for d in range(w * 7, min((w + 1) * 7, self.num_days)):
        #             weekday = date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).weekday()
        #             if weekday == 5:
        #                 sat_day = d
        #             elif weekday == 6:
        #                 sun_day = d

        #         if sat_day is None or sun_day is None:
        #             continue  # Skip weeks that don’t contain full weekend

        #         # Gather shift assignments on Saturday and Sunday
        #         sat_shifts = [
        #             self.works[(e, sat_day, s, b)]
        #             for b in employee.allowed_branches
        #             for s in range(len(SHIFT_TYPES.get(str(b), [])))
        #             if (e, sat_day, s, b) in self.works
        #         ]

        #         sun_shifts = [
        #             self.works[(e, sun_day, s, b)]
        #             for b in employee.allowed_branches
        #             for s in range(len(SHIFT_TYPES.get(str(b), [])))
        #             if (e, sun_day, s, b) in self.works
        #         ]

        #         if not sat_shifts or not sun_shifts:
        #             continue  # Nothing to enforce if no valid shifts

        #         # Create indicators for working on Saturday and Sunday
        #         worked_sat = self.model.NewBoolVar(f"worked_sat_e{e}_w{w}")
        #         worked_sun = self.model.NewBoolVar(f"worked_sun_e{e}_w{w}")

        #         self.model.AddMaxEquality(worked_sat, sat_shifts)
        #         self.model.AddMaxEquality(worked_sun, sun_shifts)

        #         # Enforce: either both worked, or neither → worked_sat == worked_sun
        #         self.model.Add(worked_sat == worked_sun)

        
        # Oude constraint
        # # night shifts alleen in combinaties inplannen
        # for e, employee in enumerate(self.employees):
        #     if not employee.works_night_shifts:
        #         continue

        #     night_shift_bools = []

        #     # Step 1: Track night shift assignments per day
        #     for d in range(self.num_days):
        #         night_vars = []

        #         for b in self.all_branches:
        #             if b not in employee.allowed_branches:
        #                 continue

        #             for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
        #                 shift_start = shift.start_time
        #                 shift_end = (shift.start_time + shift.duration) % 24

        #                 if shift_start >= 23 or shift_end < shift_start:
        #                     key = (e, d, s, b)
        #                     if key in self.works:
        #                         night_vars.append(self.works[key])

        #         if night_vars:
        #             night_shift_today = self.model.NewBoolVar(f"night_e{e}_d{d}")
        #             self.model.AddMaxEquality(night_shift_today, night_vars)
        #         else:
        #             night_shift_today = self.model.NewConstant(0)

        #         night_shift_bools.append(night_shift_today)

        #     # Step 2: Enforce "no isolated night shifts"
        #     for d in range(self.num_days):
        #         current = night_shift_bools[d]

        #         # Get neighbors if they exist
        #         neighbors = []
        #         if d > 0:
        #             neighbors.append(night_shift_bools[d - 1])
        #         if d < self.num_days - 1:
        #             neighbors.append(night_shift_bools[d + 1])

        #         if neighbors:
        #             # current => at least one neighbor is also a night shift
        #             self.model.AddBoolOr(neighbors).OnlyEnforceIf(current)

             
                        
                        
                        
                        



    # Definieer de solver
    def solve(self):
       
        # 🔹 Step 1: Compute priority-based assignment
        priority_term = sum(
            (self.employees[e].priority + 1) * sum(
                self.works[(e, d, s, b)] 
                for d in range(self.num_days) 
                for b in self.employees[e].allowed_branches
                for s in range(len(SHIFT_TYPES.get(str(b), [])))  # ✅ Use branch-specific shift count
            )
            for e in range(self.num_employees)
        )

        # ✅ Step 2: Compute Weekly Hours (Now per Branch)
        weekly_hours = self.weekly_hours  # Use precomputed values

        overwork_penalty = []
        for e in range(self.num_employees):
            # only consider vaste employees
            if self.employees[e].contract_type.lower() == "flex":
                continue # skip if flex worker
            
            for w in range(WEEKS_PER_MONTH):
                weekly_limit = int(self.employees[e].contract_hours / WEEKS_PER_MONTH)
                penalty_var = self.model.NewIntVar(0, int(weekly_limit * 0.1), f"overwork_penalty_{e}_{w}")
        
                # ✅ Penalize the cases the amount of weekly hours is exceed
                self.model.Add(penalty_var >= weekly_hours[(e, w)] - int(weekly_limit))
                overwork_penalty.append(penalty_var)
                    
        # Step 3: Introduce penalty for spread of hours assigned per week across a month
        total_hours_cap = math.ceil(
            max(emp.contract_hours for emp in self.employees) // WEEKS_PER_MONTH
             )

        max_week = self.model.NewIntVar(0, total_hours_cap, "max_week")
        min_week = self.model.NewIntVar(0, total_hours_cap, "min_week")

        for e in range(self.num_employees):
            # only consider vaste employees
            if self.employees[e].contract_type.lower() == "flex":
                continue # skip if flex worker
            
            for w in range(WEEKS_PER_MONTH):
                self.model.Add(max_week >= self.weekly_hours[(e, w)])
                self.model.Add(min_week <= self.weekly_hours[(e, w)])
                
        spread_penalty = self.model.NewIntVar(0, total_hours_cap, "spread_penalty")
        self.model.Add(spread_penalty == max_week - min_week)


        # Step 4: Introduce penalty for having dissimilar schedules per week
        weekday_counts = defaultdict(dict)
        weekday_variance_penalties = []

        for e in range(self.num_employees):
            for wd in range(7):  # 0 = Monday, ..., 6 = Sunday
                days_matching = [
                    d for d in range(self.num_days)
                    if datetime.date(REFERENCE_YEAR, REFERENCE_MONTH, d+1).weekday() == wd # use weekkay over isoweekday to align with range(7)
                ]
                
                count_var = self.model.NewIntVar(0, len(days_matching), f"weekday_{e}_{wd}")
                self.model.Add(
                    count_var == sum(
                        self.works[(e, d, s, b)]
                        for d in days_matching
                        for b in self.employees[e].allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, d, s, b) in self.works
                    )
                )
                weekday_counts[e][wd] = count_var

            # Compare every weekday pair for consistency
            for wd1 in range(7):
                for wd2 in range(wd1 + 1, 7):
                    diff = self.model.NewIntVar(0, self.num_days, f"weekday_diff_{e}_{wd1}_{wd2}")
                    self.model.AddAbsEquality(diff, weekday_counts[e][wd1] - weekday_counts[e][wd2])
                    weekday_variance_penalties.append(diff)
        
        # Step 5: Penalizes working more days within a week than expected based on the contract hours
        ROLLING_WINDOW = 5 # 5 
        shift_overload_penalties = []

        for e, employee in enumerate(self.employees):
            estimated_max_shifts = 4 if math.ceil((employee.contract_hours or 0) / 4.3 / 8) > 3 else 3
                # 3
            if estimated_max_shifts <= 0:
                continue

            for start_day in range(self.num_days - ROLLING_WINDOW + 1):
                shift_vars = []

                for d in range(start_day, start_day + ROLLING_WINDOW):
                    for b in employee.allowed_branches:
                        for s in range(len(SHIFT_TYPES.get(str(b), []))):
                            key = (e, d, s, b)
                            if key in self.works:
                                shift_vars.append(self.works[key])

                if not shift_vars:
                    continue

                # Total shifts in the rolling window
                total_shifts = self.model.NewIntVar(0, len(shift_vars), f"total_shifts_e{e}_d{start_day}")
                self.model.Add(total_shifts == sum(shift_vars))

                # Indicator: is overload active?
                overload_trigger = self.model.NewBoolVar(f"overload_trigger_e{e}_d{start_day}")

                # Define overload amount (penalty variable)
                overload = self.model.NewIntVar(0, len(shift_vars), f"overload_e{e}_d{start_day}")

                # total_shifts > allowed → trigger = 1
                self.model.Add(total_shifts > estimated_max_shifts).OnlyEnforceIf(overload_trigger)
                self.model.Add(total_shifts <= estimated_max_shifts).OnlyEnforceIf(overload_trigger.Not())

                # Link overload to how much we're over, only if triggered
                self.model.Add(overload == total_shifts - estimated_max_shifts).OnlyEnforceIf(overload_trigger)
                self.model.Add(overload == 0).OnlyEnforceIf(overload_trigger.Not())

                shift_overload_penalties.append(overload)
                
        
        # Step 5b: punish working more consecutive days than expected in contract
        consecutive_workdays_penalty = []

        # for e in range(self.num_employees):
        #     # 🧠 Calculate employee-specific maximum allowed consecutive work days
        #     employee = self.employees[e]
        #     max_consecutive_days = math.ceil(employee.contract_hours / 4.3 / 8)

        #     # 🧠 Now use employee-specific max days
        #     for d in range(self.num_days - max_consecutive_days):
        #         consecutive_days = []
        #         for offset in range(max_consecutive_days + 1):  # Check one day beyond
        #             current_day = d + offset
        #             if current_day >= self.num_days:
        #                 continue  # safeguard

        #             daily_work = sum(
        #                 self.works[(e, current_day, s, b)]
        #                 for b in employee.allowed_branches
        #                 for s in range(len(SHIFT_TYPES.get(str(b), [])))
        #                 if (e, current_day, s, b) in self.works
        #             )
        #             consecutive_days.append(daily_work)

        #         # 🌟 Introduce penalty if employee exceeds allowed work days
        #         excess = self.model.NewIntVar(0, max_consecutive_days + 1, f"excess_consec_{e}_{d}")
        #         self.model.Add(excess >= sum(consecutive_days) - max_consecutive_days)

        #         consecutive_workdays_penalty.append(excess * 50)  # Weight of penalty

                
        #Step 6: Track total assigned hours across all employees for penalization
        total_assigned_hours = []

        for e in range(self.num_employees):
            for d in range(self.num_days):
                for b in self.employees[e].allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        if (e, d, s, b) in self.works:
                            worked_hours = int(shift.worked_hours)
                            term = cp_model.LinearExpr.Term(self.works[(e, d, s, b)], worked_hours)
                            total_assigned_hours.append(term)
                            
        # Step 7: Build in blacklist into soft constraint (hard constraint prevented schedule)
        blacklist_penalties = []

        for e, employee in enumerate(self.employees):
            for d in employee.blacklisted_days:
                day_index = d - 1  # Convert to 0-based index
                if 0 <= day_index < self.num_days:
                    for b in employee.allowed_branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            key = (e, day_index, s, b)
                            if key in self.works:
                                penalty_var = self.model.NewIntVar(0, 1, f"blacklist_penalty_{e}_{day_index}_{s}_{b}")
                                # penalty_var == 1 if the shift is assigned
                                self.model.Add(penalty_var == self.works[key])
                                blacklist_penalties.append(penalty_var)
                                
        # step 8: Penalize non-consecutive shifts <= 3
        scattered_shift_penalties = []

        for e, employee in enumerate(self.employees):
            
            if not employee.unavailable_days:
                    
                for d in range(1, self.num_days - 1):  # Skip first and last day
                    # Track if the employee works on day d-1, d, d+1
                    works_prev = self.model.NewBoolVar(f"works_prev_{e}_{d}")
                    works_curr = self.model.NewBoolVar(f"works_curr_{e}_{d}")
                    works_next = self.model.NewBoolVar(f"works_next_{e}_{d}")

                    self.model.AddMaxEquality(
                        works_prev,
                        [self.works[(e, d - 1, s, b)]
                        for b in self.employees[e].allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, d - 1, s, b) in self.works]
                    )

                    self.model.AddMaxEquality(
                        works_curr,
                        [self.works[(e, d, s, b)]
                        for b in self.employees[e].allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, d, s, b) in self.works]
                    )

                    self.model.AddMaxEquality(
                        works_next,
                        [self.works[(e, d + 1, s, b)]
                        for b in self.employees[e].allowed_branches
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, d + 1, s, b) in self.works]
                    )

                    # Define a bool: 1 if [0, 1, 0] → scattered shift
                    scattered = self.model.NewBoolVar(f"scattered_{e}_{d}")

                    # Only true if prev=0 AND curr=1 AND next=0
                    self.model.AddBoolAnd([
                        works_prev.Not(),
                        works_curr,
                        works_next.Not()
                    ]).OnlyEnforceIf(scattered)

                    sandwiched = self.model.NewBoolVar(f"sandwiched_{e}_{d}")
                    self.model.AddBoolAnd([
                        works_prev,
                        works_curr.Not(),
                        works_next
                    ]).OnlyEnforceIf(sandwiched)

                    scattered_shift_penalties.append(scattered)
                    scattered_shift_penalties.append(sandwiched)
                    
            else:
                continue



       
        # step 9: penalize the weekly off-days being assigned
        free_weekday_penalties = []  # Collect soft penalties

        for e, employee in enumerate(self.employees):
            for d in range(self.num_days):
                day_of_week = datetime.date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).isoweekday()  # 1 = Monday, ..., 7 = Sunday

                if day_of_week in employee.unavailable_days:
                    for b in employee.allowed_branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            if (e, d, s, b) in self.works:
                                # Create a penalty variable for violating the unavailability
                                penalty = self.model.NewIntVar(0, 1, f"unavailable_penalty_{e}_{d}_{s}_{b}")
                                
                                # If the shift is assigned, penalty = 1
                                self.model.Add(penalty == self.works[(e, d, s, b)])
                                free_weekday_penalties.append(penalty)
                                
        # step 10: penalize the additional nv free days (niet verlof)
        free_nv_penalties = []

        for e, employee in enumerate(self.employees):
            for d in employee.free_days:  # List of 1-based day numbers
                day_index = int(d) - 1  # Convert to 0-based index

                if 0 <= day_index < self.num_days:  # Ensure within bounds
                    for b in employee.allowed_branches:
                        for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                            if (e, day_index, s, b) in self.works:
                                penalty = self.model.NewIntVar(0, 1, f"free_day_penalty_{e}_{day_index}_{s}_{b}")
                                self.model.Add(penalty == self.works[(e, day_index, s, b)])
                                free_nv_penalties.append(penalty)
                                
        # step 11: penalize deviations from the branch distributions
        branch_distribution_penalties = []

        for e, employee in enumerate(self.employees):
            desired_dist = employee.branch_distribution # e.g. {0: 50, 3: 50}

            if not desired_dist:
                continue  # No preference specified

            # Calculate total number of shifts assigned to the employee
            total_shifts = self.model.NewIntVar(0, self.num_days, f"total_shifts_e{e}")
            self.model.Add(
                total_shifts == sum(
                    self.works[(e, d, s, b)]
                    for d in range(self.num_days)
                    for b in employee.allowed_branches
                    for s in range(len(SHIFT_TYPES.get(str(b), [])))
                    if (e, d, s, b) in self.works
                )
            )

            for b, target_percent in desired_dist.items():
                b = str(b)
                # Number of shifts this employee works at branch `b`
                shift_count = self.model.NewIntVar(0, self.num_days, f"branch_shifts_{e}_{b}")
                self.model.Add(
                    shift_count == sum(
                        self.works[(e, d, s, b)]
                        for d in range(self.num_days)
                        for s in range(len(SHIFT_TYPES.get(str(b), [])))
                        if (e, d, s, b) in self.works
                    )
                )

                # Calculate expected count = total_shifts * (target_percent / 100)
                expected_scaled = self.model.NewIntVar(0, self.num_days * 1000, f"expected_scaled_{e}_{b}")
                self.model.AddMultiplicationEquality(expected_scaled, [total_shifts, int(target_percent * 10)])  # scale to 1000

                actual_scaled = self.model.NewIntVar(0, self.num_days * 1000, f"actual_scaled_{e}_{b}")
                self.model.Add(actual_scaled == shift_count * 1000)

                deviation = self.model.NewIntVar(0, self.num_days * 1000, f"deviation_{e}_{b}")
                self.model.AddAbsEquality(deviation, actual_scaled - expected_scaled)

                branch_distribution_penalties.append(deviation)
                
        # step 12: penalize flex workers from being assigned if not necessary
        flex_worker_penalties = []

        for e, employee in enumerate(self.employees):
            if employee.contract_type.lower() != "flex":
                continue  # only penalize flex workers

            for d in range(self.num_days):
                for b in employee.allowed_branches:
                    for s in range(len(SHIFT_TYPES.get(str(b), []))):
                        key = (e, d, s, b)
                        if key in self.works:
                            penalty_var = self.model.NewIntVar(0, 1, f"flex_penalty_{e}_{d}_{s}_{b}")
                            self.model.Add(penalty_var == self.works[key])
                            flex_worker_penalties.append(penalty_var)
        
        # step 13: penalize employees with a 'vast' contract for having too little hours
        underassignment_penalties = []

        for e, employee in enumerate(self.employees):
            if employee.contract_type.lower() != "vast":
                continue  # Only apply to permanent contracts

            # Track only daytime hours (e.g., shifts starting between 07:00 and 18:00)
            daytime_hours = []

            for d in range(self.num_days):
                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        if not (7 <= shift.start_time < 18):
                            continue  # ✅ Skip night shifts

                        key = (e, d, s, b)
                        if key in self.works:
                            worked_hours = int(shift.worked_hours)
                            expr = cp_model.LinearExpr.Term(self.works[key], worked_hours)
                            daytime_hours.append(expr)

            # Variable for total assigned daytime hours
            assigned_day_hours = self.model.NewIntVar(0, int(employee.net_contract_hours), f"assigned_day_hours_{e}")
            self.model.Add(assigned_day_hours == sum(daytime_hours))

            # Define threshold (e.g. 90% of contract hours)
            min_required = int(employee.net_contract_hours * 0.9)

            # Penalize shortfall
            shortfall = self.model.NewIntVar(0, int(employee.net_contract_hours), f"shortfall_{e}")
            self.model.Add(shortfall >= min_required - assigned_day_hours)

            underassignment_penalties.append(shortfall)

        
        # step 14: penalize having shifts that are far apart from meetings (such that employees work on days/shifts in line with meetings that are happening)
        meeting_distance_penalties = []

        for e, employee in enumerate(self.employees):
            for meeting in employee.meeting_times:  # Each element is a dict like {"12": "14-16"}
                for day_str, time_range in meeting.items():
                    try:
                        day = int(day_str) - 1  # Convert to 0-based index
                        if not (0 <= day < self.num_days):
                            continue

                        start_str, end_str = time_range.split("-")
                        meeting_start = float(start_str)
                        meeting_end = float(end_str)
                        meeting_midpoint = (meeting_start + meeting_end) / 2

                        gap_vars = []

                        for b in employee.allowed_branches:
                            for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                                key = (e, day, s, b)
                                if key not in self.works:
                                    continue

                                shift_midpoint = (shift.start_time + (shift.duration / 2)) % 24

                                # Define absolute time distance from shift to meeting
                                diff = self.model.NewIntVar(0, 24, f"meeting_gap_e{e}_d{day}_s{s}_b{b}")
                                raw_gap = int(abs(shift_midpoint - meeting_midpoint) * 10)  # scaled to int

                                self.model.Add(diff == raw_gap).OnlyEnforceIf(self.works[key])
                                penalty_term = cp_model.LinearExpr.Term(self.works[key], raw_gap)
                                gap_vars.append(penalty_term)

                        # Add fallback penalty if no shift is assigned that day
                        shift_vars_today = [
                            self.works[(e, day, s, b)]
                            for b in employee.allowed_branches
                            for s in range(len(SHIFT_TYPES.get(str(b), [])))
                            if (e, day, s, b) in self.works
                        ]

                        if shift_vars_today:
                            # Create a boolean variable for whether the employee is assigned any shift
                            assigned_any_shift = self.model.NewBoolVar(f"assigned_shift_e{e}_d{day}")
                            self.model.AddMaxEquality(assigned_any_shift, shift_vars_today)

                            # Create a variable to track penalty if NO shift is worked on the meeting day
                            no_shift_penalty = self.model.NewIntVar(0, 1, f"no_shift_on_meeting_e{e}_d{day}")

                            # Enforcement literals
                            has_no_shift = self.model.NewBoolVar(f"has_no_shift_e{e}_d{day}")
                            has_shift = self.model.NewBoolVar(f"has_shift_e{e}_d{day}")

                            self.model.Add(assigned_any_shift == 0).OnlyEnforceIf(has_no_shift)
                            self.model.Add(assigned_any_shift == 1).OnlyEnforceIf(has_shift)

                            self.model.Add(no_shift_penalty == 1).OnlyEnforceIf(has_no_shift)
                            self.model.Add(no_shift_penalty == 0).OnlyEnforceIf(has_shift)

                            gap_vars.append(no_shift_penalty * 20)  # Heavy penalty for missing meeting day

                        meeting_distance_penalties.extend(gap_vars)

                    except Exception as ex:
                        print(f"⚠️ Error parsing meeting time for employee {e} on day {day_str}: {ex}")
        
        
        # step 15: reduce weekends from being assgined only partly to employees                
        weekend_cohesion_penalties = []

        for e in range(self.num_employees):
            for w in range(WEEKS_PER_MONTH):
                # Compute the range of days for this week
                week_start = w * 7
                week_end = min(week_start + 7, self.num_days)

                # Extract the days corresponding to Friday (5), Saturday (6), Sunday (7)
                weekend_days = [
                    d for d in range(week_start, week_end)
                    if datetime.date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).isoweekday() in [6, 7] # Eclude 5: Friday
                ]

                # Collect variables for whether employee works on each weekend day
                work_flags = []
                for d in weekend_days:
                    works_day = self.model.NewBoolVar(f"works_e{e}_d{d}_weekend")
                    self.model.AddMaxEquality(
                        works_day,
                        [
                            self.works[(e, d, s, b)]
                            for b in self.employees[e].allowed_branches
                            for s in range(len(SHIFT_TYPES.get(str(b), [])))
                            if (e, d, s, b) in self.works
                        ]
                    )
                    work_flags.append(works_day)

                if work_flags:
                    # Sum of worked weekend days
                    weekend_sum = self.model.NewIntVar(0, len(work_flags), f"weekend_sum_e{e}_w{w}")
                    self.model.Add(weekend_sum == sum(work_flags))
                    
                    # Create intermediate boolean indicators
                    is_one = self.model.NewBoolVar(f"weekend_is_one_e{e}_w{w}")
                    # is_two = self.model.NewBoolVar(f"weekend_is_two_e{e}_w{w}")

                    self.model.Add(weekend_sum == 1).OnlyEnforceIf(is_one)
                    self.model.Add(weekend_sum != 1).OnlyEnforceIf(is_one.Not())

                    # self.model.Add(weekend_sum == 2).OnlyEnforceIf(is_two)
                    # self.model.Add(weekend_sum != 2).OnlyEnforceIf(is_two.Not())

                    # Penalize if only 1 or 2 weekend days worked
                    penalty = self.model.NewIntVar(0, 1, f"weekend_penalty_e{e}_w{w}")
                    self.model.AddMaxEquality(penalty, [is_one]) # , is_two

                    weekend_cohesion_penalties.append(penalty)

        
        # step 16: have evv people work on day shifts more often
        evv_day_shift_penalties = []

        for e, employee in enumerate(self.employees):
            if str(employee.is_evv).strip().lower() != "ja":
                continue  # Only for EVV employees

            total_hours = self.model.NewIntVar(0, int(employee.net_contract_hours), f"evv_total_hours_{e}")
            day_hours = self.model.NewIntVar(0, int(employee.net_contract_hours), f"evv_day_hours_{e}")

            total_expr = []
            day_expr = []

            for d in range(self.num_days):
                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        key = (e, d, s, b)
                        if key in self.works:
                            duration = int(shift.duration)
                            shift_end = (shift.start_time + shift.duration) % 24

                            # Accumulate all shift hours
                            total_expr.append(self.works[key] * duration)

                            # If it's a day shift (ends between 10 and 18), count it
                            if 10 <= shift_end <= 18:
                                day_expr.append(self.works[key] * duration)

            # Set the total and day shift expressions
            self.model.Add(total_hours == sum(total_expr))
            self.model.Add(day_hours == sum(day_expr))

            # Define minimum required day hours (e.g., 75%)
            min_day_hours = self.model.NewIntVar(0, int(employee.net_contract_hours), f"evv_min_day_hours_{e}")
            # Old line: self.model.Add(min_day_hours == (total_hours * 75) // 100)

            numerator = self.model.NewIntVar(0, 10000, f"evv_75pct_hours_{e}")
            self.model.Add(numerator == total_hours * 75)

            self.model.AddDivisionEquality(min_day_hours, numerator, 100)
            
            # Penalize the shortfall (how much day_hours fall short)
            shortfall = self.model.NewIntVar(0, int(employee.net_contract_hours), f"evv_day_shortfall_{e}")
            self.model.Add(shortfall >= min_day_hours - day_hours)

            evv_day_shift_penalties.append(shortfall)


        # step 17: promote even distribution between day and late shifts
        non_evv_balance_penalties = []

        for e, employee in enumerate(self.employees):
            if str(employee.is_evv).strip().lower() == "ja":
                continue  # Skip EVV employees

            day_shift_hours = []
            late_shift_hours = []

            for d in range(self.num_days):
                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        key = (e, d, s, b)
                        if key not in self.works:
                            continue

                        shift_start = shift.start_time
                        shift_end = (shift.start_time + shift.duration) % 24
                        duration = int(shift.worked_hours) # Use worked hours in order to not take into account pause times into the total body of worked hours

                        # Exclude night shifts (starting at/after 23 or ending after 23.45)
                        if shift_start >= 23 or shift_end <= shift_start or shift_end > 23.75:
                            continue

                        if shift_start < 15:
                            day_shift_hours.append(self.works[key] * duration)
                        else:
                            late_shift_hours.append(self.works[key] * duration)

            # Sum variables
            total_day = self.model.NewIntVar(0, 200, f"day_shift_hours_{e}")
            total_late = self.model.NewIntVar(0, 200, f"late_shift_hours_{e}")
            self.model.Add(total_day == sum(day_shift_hours))
            self.model.Add(total_late == sum(late_shift_hours))

            # Penalize imbalance
            imbalance = self.model.NewIntVar(0, 200, f"imbalance_shift_type_{e}")
            self.model.AddAbsEquality(imbalance, total_day - total_late)
            non_evv_balance_penalties.append(imbalance)
        
        # step 18: promote consecutive night shifts
        consecutive_night_penalties = []

        for e, employee in enumerate(self.employees):
            if not employee.works_night_shifts:
                continue

            night_shift_bools = []

            # Create Boolean variable per day indicating whether a night shift was assigned
            for d in range(self.num_days):
                night_vars = []
                for b in employee.allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):
                        start = shift.start_time
                        end = (shift.start_time + shift.duration) % 24
                        if start >= 23 or end < start:
                            key = (e, d, s, b)
                            if key in self.works:
                                night_vars.append(self.works[key])

                if night_vars:
                    shift_assigned = self.model.NewBoolVar(f"night_e{e}_d{d}")
                    self.model.AddMaxEquality(shift_assigned, night_vars)
                else:
                    shift_assigned = self.model.NewConstant(0)

                night_shift_bools.append(shift_assigned)

            # Define isolated night shifts (pattern [0, 1, 0])
            for d in range(1, self.num_days - 1):
                prev = night_shift_bools[d - 1]
                curr = night_shift_bools[d]
                next_ = night_shift_bools[d + 1]

                # Isolated = true iff [0, 1, 0]
                is_isolated = self.model.NewBoolVar(f"isolated_night_e{e}_d{d}")

                # Implement equivalence using indicator constraints:
                pattern_010 = self.model.NewBoolVar(f"pattern_010_e{e}_d{d}")

                # Define what pattern_010 means using a full logical definition (iff)
                # Define: p1 = [0, 1, 0]
                p1 = self.model.NewBoolVar(f"pattern_010_e{e}_d{d}_isolated")
                self.model.AddBoolAnd([
                    prev.Not(),
                    curr,
                    next_.Not()
                ]).OnlyEnforceIf(p1)
                self.model.AddBoolOr([
                    prev,
                    curr.Not(),
                    next_
                ]).OnlyEnforceIf(p1.Not())

                # Define: p2 = [1, 0, 1]
                p2 = self.model.NewBoolVar(f"pattern_010_e{e}_d{d}_sandwiched")
                self.model.AddBoolAnd([
                    prev,
                    curr.Not(),
                    next_
                ]).OnlyEnforceIf(p2)
                self.model.AddBoolOr([
                    prev.Not(),
                    curr,
                    next_.Not()
                ]).OnlyEnforceIf(p2.Not())

                # Combine: pattern_010 = p1 OR p2
                pattern_010 = self.model.NewBoolVar(f"undesirable_pattern_e{e}_d{d}")
                self.model.AddMaxEquality(pattern_010, [p1, p2])

                # Track for penalization
                consecutive_night_penalties.append(pattern_010)

            # Handle edge case: isolated at start ([1, 0])
            if self.num_days >= 2:
                curr = night_shift_bools[0]
                next_ = night_shift_bools[1]

                is_edge_isolated = self.model.NewBoolVar(f"isolated_start_night_e{e}_d0")
                self.model.AddBoolAnd([curr, next_.Not()]).OnlyEnforceIf(is_edge_isolated)
                self.model.AddBoolOr([curr.Not(), next_]).OnlyEnforceIf(is_edge_isolated.Not())
                consecutive_night_penalties.append(is_edge_isolated)

                # Isolated at end ([0, 1])
                prev = night_shift_bools[-2]
                curr = night_shift_bools[-1]

                is_edge_isolated = self.model.NewBoolVar(f"isolated_end_night_e{e}_d{self.num_days-1}")
                self.model.AddBoolAnd([prev.Not(), curr]).OnlyEnforceIf(is_edge_isolated)
                self.model.AddBoolOr([prev, curr.Not()]).OnlyEnforceIf(is_edge_isolated.Not())
                consecutive_night_penalties.append(is_edge_isolated)
                
        # Step 19: penalize people working consecutive weekends.      
        consecutive_weekend_penalties = []
        WEEKEND_DAYS = {5, 6}
        for e, employee in enumerate(self.employees):
            # Track weekend work per week (list of BoolVars: one per weekend)
            weekend_work_flags = []

            for w in range(WEEKS_PER_MONTH - 1):  # Stop at second-to-last week
                # Collect shifts on Saturday and Sunday for this employee
                weekend_shift_vars = []

                for d in range(w * 7, min((w + 1) * 7, self.num_days)):
                    if date(REFERENCE_YEAR, REFERENCE_MONTH, d + 1).weekday() in WEEKEND_DAYS:
                        for b in employee.allowed_branches:
                            for s in range(len(SHIFT_TYPES.get(str(b), []))):
                                key = (e, d, s, b)
                                if key in self.works:
                                    weekend_shift_vars.append(self.works[key])

                # Create a BoolVar indicating if this employee works any shift this weekend
                if weekend_shift_vars:
                    worked_weekend = self.model.NewBoolVar(f"worked_weekend_e{e}_w{w}")
                    self.model.AddMaxEquality(worked_weekend, weekend_shift_vars)
                    weekend_work_flags.append(worked_weekend)
                else:
                    # No weekend days — assume not worked
                    weekend_work_flags.append(self.model.NewConstant(0))

            # Penalize if both this weekend and the next are worked
            for w in range(len(weekend_work_flags) - 1):
                worked_this = weekend_work_flags[w]
                worked_next = weekend_work_flags[w + 1]

                both_weekends = self.model.NewBoolVar(f"consecutive_weekends_e{e}_w{w}")
                self.model.AddBoolAnd([worked_this, worked_next]).OnlyEnforceIf(both_weekends)
                self.model.AddBoolOr([worked_this.Not(), worked_next.Not()]).OnlyEnforceIf(both_weekends.Not())

                consecutive_weekend_penalties.append(both_weekends)





                  

                
        # 🔥 Minimize workload variation between weeks

        self.model.Minimize(#0.2 * priority_term  # Penalty term is strong for larger numbered employees, so the penalization should be toned down
                              overwork_penalty_factor               * sum(overwork_penalty) 
                            + spread_penalty_factor                 * spread_penalty
                            + weekday_variance_penalty_factor       * sum(weekday_variance_penalties)
                            + consecutive_workdays_penalty_factor   * sum(consecutive_workdays_penalty)
                            + shift_overload_penalty_factor         * sum(shift_overload_penalties)
                            + total_hours_penalty_factor            * sum(total_assigned_hours)
                            + blacklist_penalty_factor              * sum(blacklist_penalties)     # These two soft constraints have hard counterparts that are often hard to enforce
                            + free_weekday_penalty_factor           * sum(free_weekday_penalties)  #
                            #+ free_nv_penalty_factor                * sum(free_nv_penalties)      # Mandatory hard constraint in case of flex workers
                            + scattered_shift_penalty_factor        * sum(scattered_shift_penalties)
                            + branch_distribution_penalty_factor    * sum(branch_distribution_penalties)
                            + flex_worker_penalty_factor            * sum(flex_worker_penalties)
                            + underassignment_penalty_factor        * sum(underassignment_penalties)
                            + meeting_distance_penalty_factor       * sum(meeting_distance_penalties)
                            + weekend_cohesion_penalty_factor       * sum(weekend_cohesion_penalties)
                            + evv_day_shift_penalty_factor          * sum(evv_day_shift_penalties)
                            + non_evv_balance_penalty_factor        * sum(non_evv_balance_penalties)
                            + consecutive_night_penalty_factor      * sum(consecutive_night_penalties)
                            + consecutive_weekend_penalty_factor    * sum(consecutive_weekend_penalties)
                            )



        # Initialize scores and schedules for several seeds
        # Define your benchmark settings
        time_values = [3200]
        seeds = [3, 4]

        # Track results
        results = []
        best_score = float("inf")
        best_schedule = None

        # Loop over time limits
        for time_limit in time_values:
            print(f"\n⏱️ Evaluating time limit: {time_limit} seconds")

            for seed in seeds:
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = time_limit
                solver.parameters.num_search_workers = 8
                solver.parameters.random_seed = seed
                
                result = solver.Solve(self.model)
                
                if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    print(f"❌ No feasible solution (status: {solver.StatusName(result)})")
                    continue
                
                penalties_to_track = {"consecutive_night_penalties": consecutive_night_penalties,
                                      "blacklist_penalties": blacklist_penalties,
                                      "free_weekday_penalties": free_weekday_penalties,
                                      "consecutive_days_penalties": shift_overload_penalties,
                                      "weekend_cohesion_penalties": weekend_cohesion_penalties}
                # Track the most important penalties that are being assigned: consecutive nights and supposedly hard constraints
                for name, soft_constraint_penalty in penalties_to_track.items():
                    violations = sum(solver.Value(p) for p in soft_constraint_penalty)
                    print(f"Actual violations found for {name}:", violations)
                    print("Reported objective penalty score:", solver.ObjectiveValue())
                
            

                if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    score = solver.ObjectiveValue()
                    print(f"✅ Seed {seed} | Time {time_limit}s → Score: {score:.2f}")
                    results.append({
                        "seed": seed,
                        "time_limit": time_limit,
                        "score": score
                    })

                    # Track best result
                    if score < best_score:
                        best_score = score
                        best_schedule = solver  # Store full solver instance (or extract your best schedule here)
                else:
                    print(f"❌ Seed {seed} | Time {time_limit}s → No solution")
                    results.append({
                        "seed": seed,
                        "time_limit": time_limit,
                        "score": None
                    })
        
        #result = solver.Solve(self.model)

        #if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        #    print("Geen roosteroplossing gevonden.")
        #    return []

        
        # ✅ Collect Scheduled Shifts
        schedule = []
        for e in range(self.num_employees):
            for d in range(self.num_days):
                for b in self.employees[e].allowed_branches:
                    for s, shift in enumerate(SHIFT_TYPES.get(str(b), [])):  # ✅ Use branch-specific shifts
                        if best_schedule.Value(self.works[(e, d, s, b)]) == 1:
                            employee = self.employees[e]
                            date_str = f"{REFERENCE_YEAR}-{REFERENCE_MONTH}-{d+1:02d}"
                            func_name = ID_TO_FUNCTION.get(employee.func, "Unknown")
                            shift_name = shift.naam
                            branch_name = ID_TO_BRANCH.get(b, f"Unknown({b})")
                            schedule.append((date_str, employee.name, branch_name, shift.start_time, shift.duration, shift_name, func_name))
                            employee.assigned_hours += shift.worked_hours
                            
        return schedule, results

# ✅ Generate Schedule with OR-Tools
scheduler = OrToolsScheduler(employees, year = REFERENCE_YEAR, month = REFERENCE_MONTH)
rooster, results = scheduler.solve()

#print(pd.DataFrame(rooster))

df_rooster = pd.DataFrame(rooster)
df_rooster.columns = ["Datum", "Naam medewerker", "Afdeling", "Starttijd", "Duur", "Shift naam", "Functie"]
df_rooster.to_excel("Rooster_output_HHH.xlsx", header=["Datum", "Naam medewerker", "Afdeling", "Starttijd", "Duur", "Shift", "Functie"], index = False )


# Pivot het rooster

# Maak eerst een kolom met gecombineerde info
#df_rooster =df_rooster
df_rooster["RoosterInfo"] = df_rooster["Starttijd"].astype(str) + " - " + (df_rooster["Starttijd"] + df_rooster["Duur"]).astype(str) + " : " + df_rooster["Shift naam"]

df_rooster_pivot = df_rooster.pivot_table(
    index=["Afdeling", "Naam medewerker"],
    columns="Datum",
    values="RoosterInfo",
    aggfunc="first"
).reset_index()

# Eerst: totale duur per medewerker berekenen uit originele df
totalen = df_rooster.groupby(["Afdeling", "Naam medewerker"])["Duur"].sum().rename("Totaal uren")

# Voeg dit toe aan de gepivoteerde dataframe
df_rooster_pivot = df_rooster_pivot.merge(totalen, on=["Afdeling", "Naam medewerker"])

df_rooster_pivot.to_excel(f"Rooster_output_HHH_{REFERENCE_YEAR}_{REFERENCE_MONTH:02d}.xlsx", index = False)


