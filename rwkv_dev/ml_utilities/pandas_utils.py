from typing import Dict
from pathlib import Path
import pickle
import pandas as pd


def save_df_dict(dataframe_dict: Dict[str, pd.DataFrame], dir: Path, filename_wo_ending: str) -> None:
    save_df_dict_pickle(dataframe_dict, dir, filename_wo_ending)
    save_df_dict_xlsx(dataframe_dict, dir, filename_wo_ending)

def load_df_dict_pickle(dir: Path, filename_wo_ending: str) -> Dict[str, pd.DataFrame]:
    load_file = dir / f'{filename_wo_ending}.p'
    with load_file.open(mode='rb') as f:
        dataframe_dict = pickle.load(f)
    return dataframe_dict

def save_df_dict_pickle(dataframe_dict: Dict[str, pd.DataFrame], dir: Path, filename_wo_ending: str) -> Path:
    save_file = dir / f'{filename_wo_ending}.p'
    with save_file.open(mode='wb') as f:
        pickle.dump(dataframe_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_file


def save_df_dict_xlsx(dataframe_dict: Dict[str, pd.DataFrame], dir: Path, filename_wo_ending: str) -> Path:
    save_file = dir / f'{filename_wo_ending}.xlsx'
    with pd.ExcelWriter(save_file) as excelwriter:
        for df_name, df in dataframe_dict.items():
            df.to_excel(excel_writer=excelwriter, sheet_name=df_name)
    return save_file
