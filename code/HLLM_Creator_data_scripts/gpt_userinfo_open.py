# Copyright 2025 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, json, re
import pandas as pd
from multiprocessing import Pool
import json
import glob
import argparse
import tqdm
import time
import os
import openai
import re
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default="output")
parser.add_argument('--input_file', type=str)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--cpus', type=int, default=16)
parser.add_argument('--limit', type=int, default=300000)
parser.add_argument('--use_time', action='store_true')
args = parser.parse_args()


def extract_json_from_markdown(markdown_text):
    match = re.search(r'\{.*\}', markdown_text, flags=re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            print(f"Wrong! {json_str}")
    else:
        print("No JSON found")
    return None


base_url = os.environ['OPENAI_API_BASE_URL']
ak = os.environ['OPENAI_API_KEY']
model_name = os.environ['OPENAI_API_MODEL_NAME']
api_version = os.environ['OPENAI_API_VERSION']


def testOpenaiChatCompletions(text):
    client = openai.OpenAI(
        base_url=base_url,
        api_version=api_version,
        api_key=ak,
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "%s" % text}],
    )
    res = json.loads(completion.model_dump_json())
    return res["choices"][0]["message"]["content"]


def user_prompt_opensource(user_history):
    # Prompt in Chinese performs better
    prompt = f'''你是一位资深的图书APP运营专家，需要基于用户过去评论过的图书标题深度刻画用户兴趣和挖掘用户需求。以下是用户的历史行为（按时间由远及近排列）：
{user_history}

请按照以下步骤进行分析：
1. 仔细研究用户历史交互过的图书标题，精准概括用户的兴趣爱好，表述避免过于具体或零散。
2. 根据关键信息，深度分析并挖掘用户的长期和短期兴趣爱好。
3. 考虑用户的图书阅读偏好，例如是否倾向于购买特定类型的图书。
4. 根据以上分析，提出贴合用户兴趣和需求的推荐建议。
5. 最后，结合你之前的所有分析内容，**尽可能全面地**刻画目标用户的用户画像，并突出其中可用于推荐的用户特点。
禁止加入任何从用户行为中你无法确定的内容，回答结果按照以下json格式输出结果，所有内容用英语输出：
{{"Long-term Interest": xx,"Short-term Interest": xx,"Preference": xx,"User Needs": xx,"Delivery Suggestion": xx,"User Profile Description": xx}}
以下是你的输出：
'''
    return prompt

#     prompt = f'''You are a seasoned book APP operations expert. Your task is to deeply profile user interests and uncover user needs based on titles of books they've previously reviewed. Below is the user's historical behavior (ordered chronologically from oldest to most recent):
# {user_history}
# Please conduct your analysis following these steps:
# 1. Carefully study the book titles the user has interacted with, and accurately summarize their interests. Avoid overly specific or fragmented descriptions.
# 2. Conduct in-depth analysis to uncover the user's long-term and short-term interests based on key information.
# 3. Consider the user's book reading preferences, such as whether they tend to purchase specific types of books.
# 4. Based on the above analysis, provide recommendation suggestions that align with the user's interests and needs.
# 5. Finally, synthesize all previous analysis to comprehensively depict the target user's profile. Highlight user characteristics that can be leveraged for recommendations.
# Do not include any information that cannot be confirmed from the user's behavior. Output results strictly in the following JSON format:
# {{"Long-term Interest": xx, "Short-term Interest": xx, "Preference": xx, "User Needs": xx, "Delivery Suggestion": xx, "User Profile Description": xx}}
# Here is your output:
# '''
#     return prompt


def process_prompt(row):
    if not args.use_time:
        user_history = '\n'.join(
            [f"{x+1}: {y}" for x, y in enumerate(row['title_list'][-args.seq_len :])]
        )
    else:
        at_list = [
            datetime.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
            for ts in row['at_list'][-args.seq_len :]
        ]
        user_history = '\n'.join(
            [
                f"{x+1}: {t} reviewed {y}"
                for x, (y, t) in enumerate(
                    zip(row['title_list'][-args.seq_len :], at_list)
                )
            ]
        )
    prompt = user_prompt_opensource(user_history)
    return prompt


def process_user(row):
    retry_nums = 5
    while True:
        try:
            prompt = process_prompt(row)
            result = testOpenaiChatCompletions(prompt)
            result = extract_json_from_markdown(result.split('</think>')[-1])
            assert isinstance(result, dict), f"{result = }"
            return result
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            time.sleep(0.2)
            retry_nums -= 1
            if retry_nums <= 0:
                return None
            continue


def process(row):
    row = row.to_dict()
    user = process_user(row)
    if user is None:
        return None
    user_info = json.dumps(user, ensure_ascii=False, indent=4)
    result = {
        "uid": row['uid'],
        "user_info": user_info,
    }
    return result


def main():
    print(args)
    if args.input_file.endswith('.parquet'):
        files = sorted(glob.glob(args.input_file))
    else:
        files = sorted(glob.glob(f"{args.input_file}/*.parquet"))

    assert args.cpus < 512
    os.makedirs(args.output_file, exist_ok=True)
    for file in files:
        if os.path.exists(os.path.join(args.output_file, os.path.basename(file))):
            continue
        print(file)
        df = pd.read_parquet(file).head(args.limit)
        print(df.columns)
        print(len(df))
        with Pool(args.cpus) as pool:
            result_iterator = list(
                tqdm.tqdm(
                    pool.imap_unordered(process, [row for index, row in df.iterrows()]),
                    total=len(df),
                )
            )
        results = []
        for result in result_iterator:
            if result is None:
                continue
            results.append(result)
        df = pd.DataFrame(results)
        df.to_parquet(os.path.join(args.output_file, os.path.basename(file)))


if __name__ == '__main__':
    main()
