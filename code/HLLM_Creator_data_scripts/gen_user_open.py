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
import glob
import argparse
import tqdm
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default="output")
parser.add_argument('--input_file', nargs='+', type=str)
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--desc_len', type=int, default=4096)
parser.add_argument('--title_len', type=int, default=768)
parser.add_argument('--sample_per_split', type=int, default=100000)
args = parser.parse_args()


def gen_prompt1():
    prompt = f'''You are a senior book app operation expert, proficient in consumer psychology, and skilled at generating more clickable book titles based on user profiles and original book titles.
Your task is to rewrite a book title that is more appealing to the user based on the user profile, original book title, and book description information:
- Integrate content that users are more likely to be interested in into the book title.
- Fully consider user information to attract users' interest as much as possible, but there's no need to forcefully incorporate the user.

Here is the user profile:
'''
    return prompt


def gen_prompt2(target_title, description):
    prompt = f'''Here is the original book title:
{target_title}
Here is the book description information:
{description}
Here is the content you need to output:
'''
    return prompt


def main():
    print(args)
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)
    dfs = []
    for input_file in args.input_file:
        if input_file.endswith('.parquet'):
            files = sorted(glob.glob(input_file))
        else:
            files = sorted(glob.glob(os.path.join(input_file, '*.parquet')))
        for file in tqdm.tqdm(files):
            dfs.append(pd.read_parquet(file))
    df = pd.concat(dfs)
    print(df.columns)
    print(len(df))
    results = []
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        response = row['rewrite_title']
        row['target_title'] = row['target_title'][:args.title_len]

        prompt1 = gen_prompt1()

        user_info = row['user_info']
        description = row['target_description']
        if description:
            description = description[: args.desc_len]
        target_title = row['target_title']
        prompt2 = gen_prompt2(target_title, description)

        title_list = row['title_list'][-args.seq_len :]
        item_id_list = row['item_id_list'][-args.seq_len :]
        results.append(
            {
                "user_profile": user_info,
                "original_title": row['target_title'],
                "original_description": row['target_description'],
                "prompt1": prompt1,
                "prompt2": prompt2,
                "response": response,
                "title_list": title_list,
                "item_id_list": item_id_list,
            }
        )

    random.seed(42)
    random.shuffle(results)
    num_splits = (len(results) + args.sample_per_split - 1) // args.sample_per_split
    if num_splits == 0:
        num_splits = 1

    split_size = len(results) // num_splits
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(results)
        df = pd.DataFrame(results[start:end])
        output_file = f"{args.output_file}/p{i + 1}.parquet"
        df.to_parquet(output_file, index=False)


if __name__ == '__main__':
    main()
