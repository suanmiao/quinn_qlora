# run as a script with parameters: jsonl_path, concurrency, output_df_path 

import argparse
from quinn_eval_utils import eval_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, required=True)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--output_df_path', type=str, default=None)
    args = parser.parse_args()
    prediction_file_path = args.jsonl_path
    concurrency = args.concurrency
    output_df_path = args.output_df_path

    print(f"Start evaluation on {prediction_file_path} with concurrency {concurrency}")
    eval_jsonl(prediction_file_path, concurrency=concurrency, output_df_path=output_df_path)
    print(f"Finished evaluation on {prediction_file_path}")

