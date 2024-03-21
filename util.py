from itertools import product
import pandas as pd
import re
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm

client = OpenAI(
    base_url='https://aihubmix.com/v1',
    api_key='sk-Aqqr607kRNHcsHysA5Ca3015Eb62497995CaDf824c22E6A2')

def create_dataset(power_range, gain_S_range, gain_O_range, cost_range):
    var_ranges = {
        "power": power_range,
        "gain_S": gain_S_range,
        "gain_O": gain_O_range,
        "cost": cost_range,
    }
    combinations = list(product(*var_ranges.values()))
    df_cheat = pd.DataFrame(combinations, columns=var_ranges.keys())
    df_cheat["total"] = df_cheat["gain_S"] + df_cheat["gain_O"]
    df_cheat["isCheat"] = 0 # 0 = cheat, 1 = honest

    df_honest = pd.DataFrame(combinations, columns=var_ranges.keys())
    df_honest["total"] = df_honest["gain_S"] + df_honest["gain_O"]
    df_honest["isCheat"] = 1

    df = pd.concat([df_cheat,df_honest],axis=0)
    return df


def replace_strings (input_string, rules):
    for key, value in rules. items():
        input_string = re. sub (key, value, input_string)
        return input_string

def create_prompt(df):
    with open ('prompt.txt','r') as f:
        prompt_text = f.read()
    f.close()
    prompt_list = []
    for index,row in df.iterrows():
        isCheat_content = ''
        if (row['isCheat'] == 1):
            isCheat_content = '被抽查的玩家猜中幸运点数。'
        elif (row['isCheat'] == 0):
            isCheat_content = '被抽查的玩家猜错并谎报其猜中幸运点数。'
        # print(isCheat_content)
        replace_dict = {
        "total":str(row['total']),
        "power":str(row['power']),
        "gain_S":str(row['gain_S']),
        "gain_O":str(row['gain_O']),
        "cost":str(row['cost']),
        "isCheat":isCheat_content
        }
        # print(replace_dict)
        pattern = re.compile("|".join(replace_dict.keys()))
        # text = replace_strings(prompt_text,replace_dict)
        text = pattern.sub(lambda m: replace_dict[m.group(0)], prompt_text)
        prompt_list.append(text)
    return prompt_list
    
def create_prompt_df(data):
    df = pd.DataFrame(columns = ['option','prompt'])

    with open ('prompt_constant.txt','r') as f:
        prompt_constant = f.read()
    f.close()

    with open ('prompt_option.txt','r') as f:
        prompt_accept = f.readlines(1)[0]
    f.close()
    # print(prompt_accept)
    prompt_list = []
    for index,row in data.iterrows():
        isCheat_content = ''
        if (row['isCheat'] == 1):
            isCheat_content = '被抽查的玩家猜中幸运点数。'
        elif (row['isCheat'] == 0):
            isCheat_content = '被抽查的玩家猜错并谎报其猜中幸运点数。'
        # print(isCheat_content)
        replace_dict = {
        "total":str(row['total']),
        "power":str(row['power']),
        "gain_S":str(row['gain_S']),
        "gain_O":str(row['gain_O']),
        "cost":str(row['cost']),
        "isCheat":isCheat_content
        }
        # print(replace_dict)
        pattern = re.compile("|".join(replace_dict.keys()))
        # text = replace_strings(prompt_text,replace_dict)
        text_constant = pattern.sub(lambda m: replace_dict[m.group(0)], prompt_constant)
        option_accept = pattern.sub(lambda m: replace_dict[m.group(0)], prompt_accept)
        option_reject = '如果你拒绝，你和TA会获得0金币，第三方会损失0金币'
        
        text_accept = text_constant + '\n' +option_accept
        text_reject = text_constant + '\n' +option_reject

        df.loc[len(df.index)] = ['accept',text_accept]
        df.loc[len(df.index)] = ['reject',text_reject]

        # prompt_list.append(d)
    return df

def get_embeddings(text):
    response = client.embeddings.create(
            model='text-embedding-3-large',
            input=text,
            dimensions=1536)
    return response.data[0].embedding

# def text_embedding(df):
#     df['embedding'] = df['prompt'].apply(get_embeddings)
#     return df


def text_embedding(max_worker,df):
    max_worker = max_worker
    # df.loc['embedding'] = ''
    embedding_result = []
    # with tqdm(total = len(df)) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(get_embeddings,texts) for texts in df['prompt']]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            # df['embedding'] = [future.result()]
            embedding_result.append(future.result())
        # for future in concurrent.futures.as_completed(futures):
            # embedding_result.append(future.result().data[0].embedding)
            # df.iloc['embedding'] = [future.result().data[0].embedding]
            # df.loc['embedding'] = embedding_result
                # pbar.update()
    return embedding_result

def behavior_embedding():
    