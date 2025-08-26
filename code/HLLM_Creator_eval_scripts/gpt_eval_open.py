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
import argparse
import tqdm
import time
import openai
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--key1', type=str, default="original_title")
parser.add_argument('--key2', type=str, default="llm_output")
parser.add_argument('--v', action='store_true')
parser.add_argument('--mprun', action='store_true')
parser.add_argument('--cpus', type=int, default=16)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--limit', type=int, default=500)
args = parser.parse_args()

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


def gen_prompt(row, a, b):
    try:
        selling = row['original_description']
    except:
        selling = row['target_description']
    selling_point_str = selling if selling else None
    try:
        user_profile = row['user_profile']
    except:
        user_profile = row['user_info']
    if 'title_list' in row:
        user_history = '\n'.join(
            [f"{x+1}: {y}" for x, y in enumerate(row['title_list'][-args.seq_len :])]
        )
    else:
        user_history = None

    prompt = f"""You are a seasoned book APP operations expert proficient in consumer psychology, highly skilled at capturing user interest preferences.
Below I will provide book titles the user has previously commented on (ordered chronologically from oldest to most recent) along with the user profile. Please deeply understand the user's interests based on this information.
I will also provide the target book's description.
Additionally, I will provide two book titles a and b. Please comprehensively evaluate all information to determine which title the user would likely be more interested in, and respond with (a or b or same).
Input details:
Book description: {selling_point_str}
User's historical behavior: {user_history}
User profile: {user_profile}
Title a: {a}
Title b: {b}
Note: Output must contain ONLY the answer (a or b or same).
Output: """
    return prompt


def process(row):
    retry = 5
    a, b = row[args.key1], row[args.key2]
    prompt1, prompt2 = gen_prompt(row, a, b), gen_prompt(row, b, a)
    while True:
        try:
            result1 = testOpenaiChatCompletions(prompt1).split('</think>')[-1].strip()
            result2 = testOpenaiChatCompletions(prompt2).split('</think>')[-1].strip()
            assert result1 in ['a', 'b', 'same'] and result2 in [
                'a',
                'b',
                'same',
            ], f"{result1} {result2}"
        except Exception as e:
            retry -= 1
            if retry == 0:
                return 'same'
            print(e)
            time.sleep(5)
            continue
        break
    if result1 == result2 or result1 == 'same' or result2 == 'same':
        return 'same'
    else:
        return result1


def main():
    df = pd.read_parquet(args.input_file).head(args.limit)
    tot, a, b, same = 0, 0, 0, 0
    results = []
    if args.mprun:
        with Pool(args.cpus) as pool:
            result_iterator = list(
                tqdm.tqdm(
                    pool.imap_unordered(process, [row for index, row in df.iterrows()]),
                    total=len(df),
                )
            )
        for result in result_iterator:
            tot += 1
            if args.dump_result:
                results.append(result)
            else:
                if result == 'a':
                    a += 1
                elif result == 'b':
                    b += 1
                else:
                    same += 1
                if tot % 10 == 0:
                    print(f"{tot = } {a = } {b = } {same = }")
    else:
        for idx, row in df.iterrows():
            result = process(row)
            if args.v:
                print(result)
            tot += 1
            if result == 'a':
                a += 1
            elif result == 'b':
                b += 1
            else:
                same += 1
            if tot % 10 == 0:
                print(f"{tot = } {a = } {b = } {same = }")
    print(f"{tot = } {a = } {b = } {same = }")


if __name__ == '__main__':
    main()
