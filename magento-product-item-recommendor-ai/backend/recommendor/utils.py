import pandas as pd

def load_pricelist(path: str):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise Exception(f"Price list not found at {path}")