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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str)
parser.add_argument("--model_paths", type=str, nargs="+")
parser.add_argument("--model_keys", type=str, nargs="+")
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

base_df = pd.read_parquet(args.base_path)
print(len(base_df))
for model_path, model_key in zip(args.model_paths, args.model_keys):
    model_df = pd.read_parquet(model_path)
    base_df = pd.concat(
        [base_df, model_df[['llm_output']].rename(columns={'llm_output': model_key})],
        axis=1,
    )

print(len(base_df))
base_df.to_parquet(args.output_path)
