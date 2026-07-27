import pandas as pd


def create_sample(
    df: pd.DataFrame,
    sample_size: int = 12000
) -> pd.DataFrame:
    """
    Create stratified sample preserving product distribution.
    """

    sample_df = pd.concat(
        [
            group.sample(
                n=int(sample_size * len(group) / len(df)),
                random_state=42
            )
            for _, group in df.groupby("Product")
        ]
    ).reset_index(drop=True)

    return sample_df
