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
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--cpus', type=int, default=16)
parser.add_argument('--limit', type=int, default=300000)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--file_limit', type=int, default=30000)
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


def construct_rewrite_prompt_withuser_opensource(
    user_profile, title, description, title_list_str
):
    # Prompt in Chinese performs better
    prompt = f'''你是一位资深的图书APP运营专家，精通消费者心理学，擅长根据用户画像和原始书名生成更吸引用户点击的书名。
你的任务是基于用户画像、历史点击的图书、原始书名和书籍描述信息，改写出更吸引该用户的书名。
通过深刻理解用户画像内容，从而灵活融入用户信息，生成个性化内容。
要求：
- 完全基于提供的用户信息、历史点击的图书、原始书名和书籍描述信息，不要引入任何其他新信息，禁止无中生有。
- 充分考虑用户信息，尽可能吸引用户兴趣，但不必强行融入用户。
- 在原书名基础上额外添加至多10个字，禁止出现用户年龄、地点相关内容，生成的书名生动、流畅、不僵硬。

请按照以下步骤进行生成：
- 判断用户画像中适合融入的信息。
- 提取用户可能更感兴趣的内容（可以为空字符串）。
- 将以上信息自然地融入，输出必须为英语，输出格式： {{"Suitable User Info": "xx", "Content Users May Be Interested In": "xx", "Book Title Combined with Users": "xx"}}

以下是目标用户画像：
{user_profile}
以下是用户历史点击过的图书：
{title_list_str}
以下是原始书名：
{title}
以下是对书的描述：
{description}
以下是你生成的文案，不要输出任何其他内容：
'''
    return prompt
#     prompt = f'''You are a seasoned book APP operations expert proficient in consumer psychology, skilled at generating book titles that attract more clicks based on user profiles and original titles.
# Your task is to rewrite a more appealing book title for this specific user based on their profile, historically clicked books, original title, and book description.
# By deeply understanding the user profile content, flexibly incorporate user-specific information to create personalized content.

# Requirements:
# - Strictly base your output on the provided user information, historical clicked books, original title, and book description. Do not introduce any new information or fabricate details.
# - Fully consider user information to maximize appeal to user interests, but avoid forced integration of user elements.
# - Add at most 10 extra characters to the original title. Prohibit any mention of user age or location. Ensure generated titles are vivid, fluent, and natural.

# Follow these generation steps:
# - Identify suitable user profile information for integration
# - Extract content that may interest the user (can be empty string)
# - Naturally incorporate the above information. Output must be in English using this format: {{"Suitable User Info": "xx", "Content Users May Be Interested In": "xx", "Book Title Combined with Users": "xx"}}

# Below is the target user profile:
# {user_profile}
# Below are books the user historically clicked:
# {title_list_str}
# Below is the original book title:
# {title}
# Below is the book description:
# {description}
# Below is your generated copy (output ONLY the specified JSON, no additional content):
# '''
#     return prompt


def get_prompt(row, user_fn=None):
    user_info = row.get('user_info', None)
    title = row['target_title']
    description = row['target_description']
    if not args.use_time:
        title_list_str = '\n'.join(
            [f"{x+1}: {y}" for x, y in enumerate(row['title_list'][-args.seq_len :])]
        )
    else:
        at_list = [
            datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            for ts in row['at_list'][-args.seq_len :]
        ]
        title_list_str = '\n'.join(
            [
                f"{x+1}: {t} reviewed {y}"
                for x, (y, t) in enumerate(
                    zip(row['title_list'][-args.seq_len :], at_list)
                )
            ]
        )
        prompt = construct_rewrite_prompt_withuser_opensource(
            user_info, title, description, title_list_str
        )
    return prompt


def process(row):
    row = row.to_dict()
    retry_nums = 5
    while True:
        try:
            prompt = get_prompt(row)
            if prompt is None:
                return None
            result = testOpenaiChatCompletions(prompt).split('</think>')[-1].strip()
            result = extract_json_from_markdown(result.split('</think>')[-1])[
                'Book Title Combined with Users'
            ]
        except Exception as e:
            print(e)
            import traceback

            print(traceback.format_exc())
            time.sleep(1)
            retry_nums -= 1
            if retry_nums <= 0:
                return None
            continue
        break
    row.pop('rewrite_title', None)
    results = {**row, "rewrite_title": result, "user_info": row.get('user_info', None)}
    return results


def main():
    print(args)
    if args.input_file.endswith('.parquet'):
        files = sorted(glob.glob(args.input_file))
    else:
        files = sorted(glob.glob(os.path.join(args.input_file, '*.parquet')))
    files = files[: args.file_limit]
    print(len(files))

    assert args.cpus < 512
    os.makedirs(args.output_file, exist_ok=True)
    for file in files:
        if os.path.exists(os.path.join(args.output_file, os.path.basename(file))):
            continue
        print(file)
        df = pd.read_parquet(file).head(args.limit)
        print(df.columns, len(df))
        with Pool(args.cpus) as pool:
            result_iterator = list(
                tqdm.tqdm(
                    pool.imap_unordered(process, [row for index, row in df.iterrows()]),
                    total=len(df),
                )
            )
        results = []
        for result in result_iterator:
            if result is not None:
                results.extend(result)
        print(len(df), len(results))
        df = pd.DataFrame(results)
        df.to_parquet(os.path.join(args.output_file, os.path.basename(file)))


if __name__ == '__main__':
    main()
