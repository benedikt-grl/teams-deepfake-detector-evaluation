import boto3
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path


def download_file(bucket, key, output_dir):
    local_path = Path(output_dir) / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return

    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, str(local_path))
    except Exception as e:
        print(f"[ERROR] Failed to download {bucket}/{key}: {e}")


def process_row(row, output_dir):
    return download_file(row['s3_bucket'], row['s3_object_key'], output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download S3 files from a CSV manifest.")
    parser.add_argument('--csv_file', help='CSV file with the selected videos', default="assets/2025_07_16-selected_videos.csv")
    parser.add_argument('--output_dir', help='Directory to download the files into', default=r"P:\GetRealLabs\grl-mleng-dsfactory-training-video-data")
    args = vars(parser.parse_args())

    df = pd.read_csv(args["csv_file"])

    if 's3_bucket' not in df.columns or 's3_object_key' not in df.columns:
        raise ValueError("CSV must contain 'bucket' and 'key' columns.")

    worker_func = partial(process_row, output_dir=args["output_dir"])
    num_workers = min(len(df), cpu_count())
    with Pool(num_workers) as pool:
        with tqdm(total=len(df)) as pbar:
            for _ in pool.imap_unordered(worker_func, [row for _, row in df.iterrows()]):
                pbar.update(1)
