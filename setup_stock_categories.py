import json
import argparse
import sys
import os
import shutil

parser = argparse.ArgumentParser(
    description="Determine handling of fetched data.")
parser.add_argument("--clean", action="store_true",
                    default=False, dest="clean")


def wrap(id, title, tickers, image_path, description):  # type: ignore
    return {"id": id, "title": title, "tickers": tickers, "image_path": image_path, "description": description}


json_bundles = []
json_bundles.append(wrap(  # type: ignore
    id="green_energy_sector",
    title="Green Energy Sector",
    tickers=["Plug Power Inc", "Enphase ENergy Inc",
             "Verbund AG", "DAQO New Energy ADR Representing",
             "Siemens Gamesa Renewable Ene"],
    image_path="resources/energy_category_thumbnail.jpg",
    description="No one likes polluting the earth, but our energy addition is hard to break. Clean"
))

CATEGORY_DATA_LOCATION = "resources/category_info/"


def main() -> None:
    if not os.path.exists(CATEGORY_DATA_LOCATION):
        os.mkdir(CATEGORY_DATA_LOCATION)
    for bundle in json_bundles:
        filename = "%s%s.json" % (CATEGORY_DATA_LOCATION, bundle["id"])
        with open(filename, 'w') as outjson:
            json.dump(bundle, outjson)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.clean:
        if os.path.exists(CATEGORY_DATA_LOCATION):
            shutil.rmtree(CATEGORY_DATA_LOCATION)
    else:
        main()
