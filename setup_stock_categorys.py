import json


def wrap(title, tickers, image_path, description):
    return {"title": title, "tickers": tickers, "image_path": image_path, "description": description}


json_bundles = []
json_bundles.append(wrap(
    title="Green Energy Sector",
    tickers=["Plug Power Inc", "Enphase ENergy Inc",
             "Verbund AG", "DAQO New Energy ADR Representing",
             "Siemens Gamesa Renewable Ene"],
    image_path="resources/energy_category_thumbnail.jpg",
    description="No one likes polluting the earth, but our energy addition is hard to break. Clean"
))
print(json_bundles[0])


def main() -> None:
    pass


if __name__ == "__main__":
    main()
