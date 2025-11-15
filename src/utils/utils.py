import pandas as pd
import numpy as np

from src.logger.logger import logging

def remove_missing_rooms(df: pd.DataFrame):
    pass

def rename_columns(df: pd.DataFrame)  -> pd.DataFrame:
    try:
        df.rename(columns={
        "css-1hidc9c": "rooms",
        "css-gkudnx": "furnish",
        "T_textContainerStyle": "builtup_area",
        "css-1ty5xzi": "address",
        "css-10rvbm3": "rent",
        "_26jk66": "nearby",
        "T_overviewStyle": "details",
        "T_sectionStyle": "furnish_detail",
        "T_sectionStyle (2)": "amenities",
        "T_highlightContainer": "highlights",
        "T_arrangeElementsInLine href": "link"
        }, inplace=True)

        logging.debug("Renamed the data")

        return df
    
    except Exception as e:
        logging.error('Unexpected error occurred while renaming the data: %s', e)
        raise


    def create_new_df(data: pd.DataFrame) -> pd.DataFrame :
        try :
            rooms = [float(df.rooms.iloc[i].split()[0]) for i in range(len(df))]
            logging.debug("created rooms list")


        except Exception as e:
            logging.error('Unexpected error occurred while creating rooms list: %s', e)
            raise


        try :
            builtup_area = [float(df.builtup_area.iloc[i].split()[0]) for i in range(len(df))]
            logging.debug("created builtup_area list")

        except Exception as e:
            logging.error('Unexpected error occurred while creating builtup_area list: %s', e)
            raise

        try :
            rents = [float(''.join((df.rent.iloc[i].split()[0]).split(','))) for i in range(len(df))]
            logging.debug("created rents list")

        except Exception as e:
            logging.error('Unexpected error occurred while creating rents list: %s', e)
            raise

            
        try :
            pool = [1 if 'Pool' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]
            gym = [1 if 'Gym' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]
            lift = [1 if 'Lift' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]
            backup = [1 if 'Backup' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]
            intercom = [1 if 'Intercom' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]
            garden = [1 if 'Garden' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]
            sports = [1 if 'Sports' in df.amenities.iloc[i].split() else 0 for i in range(len(df))]

            logging.debug("created rents list")

        except Exception as e:
            logging.error('Unexpected error occurred while creating rents list: %s', e)
            raise
    
        