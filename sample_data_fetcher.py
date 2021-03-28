import yfinance as wife
from pprint import pprint
import itertools
import argparse
import sys
import os
import shutil

# microsoft, google, tesla, taiwan semi conductor, Citi group, Bumble, Apple
tickers_ = ["MSFT", "GOOGL", "TSLA", "TSM", "C", "CSCO", "AAPL"]
FILE_PREFIX = "_hist_data.csv"
DEFAULT_DATA_FILE = "__data_file"
parser = argparse.ArgumentParser(
    description="Determine handling of fetched data.")
parser.add_argument("--clean", action="store_true",
                    default=False, dest="clean")
parser.add_argument(
    "--data_folder", default=DEFAULT_DATA_FILE, dest="data_folder")


def GetData(fin):  # type: ignore
    ticker = wife.Ticker(fin)
    return ticker.history(period="max")


def main() -> None:
    if not os.path.exists(f"{args.data_folder}/"):
        os.mkdir(f"{args.data_folder}/")
    for ticker, data in zip(tickers_, list(map(GetData, tickers_))):
        data.to_csv(f"{args.data_folder}/{ticker}{FILE_PREFIX}")
        # with open(f"{args.data_folder}{ticker}{FILE_PREFIX}", 'x') as file:
        #     file.write(data)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.clean:
        for ticker in tickers_:
            # fname = f"{ticker}{FILE_PREFIX}"
            if os.path.exists(args.data_folder):
                shutil.rmtree(args.data_folder)
    else:
        main()
