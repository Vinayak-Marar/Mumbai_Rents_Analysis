import pandas as pd
import numpy as np

import re

from src.logger.logger import logging



def extract_nearby_categories(raw_text):
    text = raw_text.lower()
    nearby = {
        'school': 1 if 'school' in text else 0,
        'hospital': 1 if 'hospital' in text else 0,
        'metro_station': 1 if 'metro' in text else 0,
        'airport': 1 if 'airport' in text else 0,
        'railway' : 1 if 'railway' in text else 0,
        'restaurant': 1 if 'restaurant' in text else 0,
        'mall' : 1 if 'mall' in text else 0,
        'bus_stand': 1 if 'bus' in text else 0
    }
    return nearby


def extracting_nearby_list(df):
    school_n = []
    university_n = []
    airport_n = []
    hospital_n = []
    metro_station_n = []
    railway_n = []
    restaurant_n = []
    mall_n = []
    bus_stand_n = []

    for i in range(len(df)):
        info = extract_nearby_categories(df.nearby.iloc[i])
        school_n.append(info['school'])
        hospital_n.append(info['hospital'])
        airport_n.append(info['airport'])
        metro_station_n.append(info['metro_station'])
        railway_n.append(info['railway'])
        restaurant_n.append(info['restaurant'])
        mall_n.append(info['mall'])
        bus_stand_n.append(info['bus_stand'])

    school_h= []
    airport_h = []
    hospital_h = []
    metro_station_h = []
    railway_h = []
    restaurant_h = []
    mall_h = []
    bus_stand_h = []

    for i in range(len(df)):
        
        info = extract_nearby_categories(df.highlights.iloc[i])
        school_h.append(info['school'])
        hospital_h.append(info['hospital'])
        airport_h.append(info['airport'])
        metro_station_h.append(info['metro_station'])
        railway_h.append(info['railway'])
        restaurant_h.append(info['restaurant'])
        mall_h.append(info['mall'])
        bus_stand_h.append(info['bus_stand'])

    school = [1 if (x or y) else 0 for x,y in zip(school_h,school_n)]
    airport = [1 if (x or y) else 0 for x,y in zip(airport_h,airport_n)]
    hospital = [1 if (x or y) else 0 for x,y in zip(hospital_h,hospital_n)]
    metro_station = [1 if (x or y) else 0 for x,y in zip(metro_station_h,metro_station_n)]
    railway = [1 if (x or y) else 0 for x,y in zip(railway_h,railway_n)]
    restaurant = [1 if (x or y) else 0 for x,y in zip(restaurant_h,restaurant_n)]
    mall = [1 if (x or y) else 0 for x,y in zip(mall_h,mall_n)]
    bus_stand = [1 if (x or y) else 0 for x,y in zip(bus_stand_h,bus_stand_n)]

    return school, airport, hospital, metro_station, railway, restaurant, mall, bus_stand

def extract_info(lst):
    info = {}

    def safe_index(word):
        try:
            return lst.index(word)
        except ValueError:
            return -1

    def get_next(lst, i, offset=1, default=np.nan):
        try:
            return lst[i + offset]
        except IndexError:
            return default

    def to_float(x):
        if not x:
            return 0.0
        try:
            return float(str(x).replace(',', '').strip())
        except:
            nums = re.findall(r'\d+\.?\d*', str(x))
            return float(nums[0]) if nums else 0.0
        
    def to_int(x):
        if not x:
            return 0
        try:
            return int(str(x).replace(',', '').strip())
        except:
            nums = re.findall(r'\d+\.?\d*', str(x))
            return int(nums[0]) if nums else 0

    # ---- Bathrooms ---- #
    i = safe_index('Bathrooms')
    info['bathrooms'] = to_int(get_next(lst, i, 1)) if i != -1 else 0

    # ---- Balcony ---- #
    i = safe_index('Balcony')
    if i != -1:
        val = get_next(lst, i, 1, '').lower()
        info['balcony'] = 0 if val == 'no' else to_int(val)
    else:
        info['balcony'] = 0

    # ---- Floor extraction ---- #
    i = safe_index('Floor')
    if i != -1:
        # take a larger window around the 'Floor' token and normalize weird spaces/chars
        start = max(0, i - 3)
        end = i + 12
        floor_text = " ".join(lst[start:end]).lower()
        floor_text = re.sub(r'[\u00A0\u200B]', ' ', floor_text)   # NBSP, zero-width spaces
        floor_text = re.sub(r'[\:\-\—]', ' ', floor_text)        # normalize some separators
        floor_text = re.sub(r'\s+', ' ', floor_text).strip()
    
        current = None
        total = None
    
        # 1) Common numeric forms: "8 of 40", "8 out of 40", "8/40", "8 / 40"
        m = re.search(r'(\d+)\s*(?:of|out\s+of|/|/)\s*(\d+)', floor_text)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
        else:
            # 2) Ground floor variants: "ground of 6", "ground floor of 6", "ground floor - of 6"
            m = re.search(r'ground(?:\s+floor)?(?:\s+number)?\s*(?:of|out\s+of|/)\s*(\d+)', floor_text)
            if m:
                current = 0
                total = int(m.group(1))
            else:
                # 3) Abbreviations like "GF of 6" or "G.F. of 6"
                m = re.search(r'\b(?:gf|g\.f\.|ground)\b[^\d]{0,6}?(?:of|out\s+of|/)\s*(\d+)', floor_text)
                if m:
                    current = 0
                    total = int(m.group(1))
                else:
                    # 4) Some noisy forms: "... ground floor - 6 floors" -> try to capture last number as total
                    m = re.search(r'ground(?:\s+floor)?[^\d]{0,8}(\d+)', floor_text)
                    if m:
                        current = 0
                        total = int(m.group(1))
    
        if current is not None and total is not None:
            info['current_floor'] = current
            info['total_floors'] = total
        else:
            info['current_floor'] = info['total_floors'] = 0
    else:
        info['current_floor'] = info['total_floors'] = 0
    
    # ---- Lease type ---- #
    i = safe_index('Lease')
    if i != -1:
        j = i + 2
        lease_parts = []
        while j < len(lst) and lst[j] not in ['Age', 'of', 'property']:
            lease_parts.append(lst[j])
            j += 1
        info['lease_type'] = ' '.join(lease_parts).replace('/', '').strip()
    else:
        info['lease_type'] = np.nan

    # ---- Age of property ---- #
    i = safe_index('Age')
    if i != -1:
        info['property_age'] = to_float(get_next(lst, i, 3))
    else:
        info['property_age'] = 0.0

    # ---- Parking ---- #
    info['covered_parking'] = 0
    info['open_parking'] = 0
    for j, word in enumerate(lst):
        if word.lower() == 'covered':
            info['covered_parking'] = to_int(get_next(lst, j, -1))
        elif word.lower() == 'open':
            info['open_parking'] = to_int(get_next(lst, j, -1))

    # ---- Facing ---- #
    i = safe_index('facing')
    info['facing'] = get_next(lst, i, 1, None) if i != -1 else "None"

    # ---- Gas pipeline ---- #
    i = safe_index('Pipeline')
    if i != -1:
        val = get_next(lst, i, 1, '').lower()
        info['gas_pipeline'] = 1 if val == 'yes' else 0
    else:
        info['gas_pipeline'] = 0

    # ---- Gated community ---- #
    i = safe_index('Community')
    if i != -1:
        val = get_next(lst, i, 1, '').lower()
        info['gated_community'] = 1 if val == 'yes' else 0
    else:
        info['gated_community'] = 0

    return info





def normalize_lease_type(x):
    if pd.isna(x):
        return 'Bachelor Company Family'
    # Split by spaces, remove empty, unique, and sort alphabetically
    parts = sorted(set(x.split()))
    return ' '.join(parts)






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


def create_new_df(df: pd.DataFrame) -> pd.DataFrame :
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

        logging.debug("created amenities list")

    except Exception as e:
        logging.error('Unexpected error occurred while creating amenities list: %s', e)
        raise

    try:
        bathroom =[]
        balcony=[]
        current_floor= []
        total_floors = []
        lease_type=[]
        property_age = []
        covered_parking = []
        open_parking = []
        facing = []
        gas_pipeline=[]
        gated_community = []

        for i in range(len(df)):
            text = df.details.iloc[i]
            clean_text = clean_text = re.sub(r'^.*?Security', 'Security', text, flags=re.S)
            info = extract_info(clean_text.split())
            bathroom.append(info['bathrooms'])
            balcony.append(info['balcony'])
            current_floor.append(info['current_floor'])
            total_floors.append(info['total_floors'])
            lease_type.append(info['lease_type'])
            property_age.append(info['property_age'])
            covered_parking.append(info['covered_parking'])
            open_parking.append(info['open_parking'])
            facing.append(info['facing'])
            gas_pipeline.append(info['gas_pipeline'])
            gated_community.append(info['gated_community'])


        lease_type = [normalize_lease_type(x) for x in lease_type]

        logging.info('Created basic informational columns list')

    except Exception as e:
        logging.error('Unexpected error occurred while creaing basic informational columns list: %s', e)
        raise  

    try:
        school, airport, hospital, metro_station, railway, restaurant, mall, bus_stand = extracting_nearby_list(df)
        
        logging.info('Created nearby lists')

    except Exception as e:
        logging.error('Unexpected error occurred while creaing basic nearby list: %s', e)
        raise
        
    try:
        all_list = {'link': df.link.to_list(),
        'builtup_area': builtup_area,
        'rooms': rooms,
        'furnish': df.furnish.to_list(),
        'address': df.address.to_list(),
        'bathrooms': bathroom,
        'balcony': balcony,
        'facing': facing,
        'gas_pipline': gas_pipeline,
        'gated_community': gated_community,
        'swimming_pool': pool,
        'gym': gym,
        'intercom': intercom,
        'power_backup': backup,
        'garden': garden,
        'sports': sports,
        'current_floor': current_floor,
        'total_floor' : total_floors,
        'lease_type' : lease_type,
        'property_age': property_age,
        'covered_parking': covered_parking,
        'open_parking' : open_parking,
        'school/university': school,
        'airport': airport,
        'bus_stop': bus_stand,
        'railway': railway,
        'mall': mall,
        'metro_station': metro_station,
        'hospital': hospital,
        'restaurant': restaurant,
        'rent' : rents}

        data = pd.DataFrame(all_list)

        logging.info("Created Dataframe")

    except Exception as e:
        logging.error('Unexpected error occurred while creating dataframe : %s', e)
        raise

    return data