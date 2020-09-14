import os
import datetime
import time
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import shutil
import backoff

import rasterio
from time import sleep
from rasterio import plot
from shapely.geometry import MultiPolygon, shape, Point
from shapely_geojson import dumps

from pathlib import Path
from pprint import pprint
from zipfile import ZipFile

from planet import api
import planet
from planet.api import filters
from tqdm.auto import tqdm

from parameters import *


DOWNLOAD_PATH = os.path.join(os.getcwd(), 'downloads')
OUT_PIKL_PATH = os.path.join(os.getcwd(), 'searches')
LOG_PATH = os.path.join(os.getcwd(), 'logs')

Path(OUT_PIKL_PATH).mkdir(parents=True, exist_ok=True)
Path(DOWNLOAD_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)


def save_thumb(metadata_df):
    """ From the metadata dataframe, save the thumbnail
    in the corresponding folder:
    
    Args:
        metadata_df (pd.DataFrame)
        
    Return:
        stores thumbnails in folder
    """
    session = requests.Session()
    session.auth = (PLANET_API_KEY, '')
    auth = session.auth
    
    for index, row in metadata_df.iterrows():
        url = row.thumbnail
        date = row.date
        item_type = row.item_type
        cloud_cover = row.cloud_cover
        id_ = row.id
        sample_id = row.sample_id
        
        thumb_name = f'it{item_type}_cc{cloud_cover}_y{date.year}m{date.month}_{id_}.jpg'
        
        thumb_path = os.path.join(os.getcwd(),'thumbs', 
                                  str(sample_id), 
                                  str(date.year))
        
        Path(thumb_path).mkdir(parents=True, exist_ok=True)

        r = requests.get(url, auth=auth, stream=True)
        if r.status_code == 200:
            with open(os.path.join(thumb_path, thumb_name), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

def build_request(aoi_geom, start_date, stop_date, cloud_cover=100):
    """build a data api search request for PS imagery.
    
    Args:
        aoi_geom (geojson): 
        start_date (datetime.datetime)
        stop_date (datetime.datetime)
    
    Returns:
        Request
    """
    
    query = filters.and_filter(
        filters.geom_filter(aoi_geom),
        filters.range_filter('cloud_cover', lte=cloud_cover),
        filters.date_range('acquired', gt=start_date),
        filters.date_range('acquired', lt=stop_date)
    )
    
    # Skipping REScene because is not orthorrectified and 
    # cannot be clipped.
    
    return filters.build_search_request(query, [
        'PSScene3Band', 
        'PSScene4Band', 
        'PSOrthoTile',
        'REOrthoTile',])

    
@backoff.on_exception(backoff.expo, planet.api.exceptions.OverQuota, max_time=360)
def get_items(id_name, request, client):
    """ Get items using the request with the given parameters
           
    """
    result = client.quick_search(request)
 
    items_pages = []
    limit_to_x_pages = None
    for page in result.iter(limit_to_x_pages):
        items_pages.append(page.get())

    items = [item for page in items_pages for item in page['features']]
    
    
    return (id_name, items)
    
def get_dataframe(items):
    
    items_metadata = [(f['properties']['acquired'],
                     f['id'], 
                     f['properties']['item_type'],
                     f['_links']['thumbnail'],
                     f['_permissions'],
                     f['geometry'],
                     f['properties']['cloud_cover'],
                     f
                    ) for f in items[1] ]
    
    # Store into dataframe
    df = pd.DataFrame(items_metadata)
    df[0] = pd.to_datetime(df[0])
    df.columns=[
        'date', 
        'id', 
        'item_type', 
        'thumbnail', 
        'permissions', 
        'footprint', 
        'cloud_cover', 
        'metadata'
    ]
    df['sample_id'] = items[0]
    df.sort_values(by=['date'], inplace=True)
    df.reset_index()
    
    return df
    
def add_cover_area(metadata_df, sample_df):
    
    for idx, row in metadata_df.iterrows():
        
        g1 = sample_df.at[row.sample_id, 'geometry'] # sample geometry
        g2 = shape(row.footprint) # footprint geometry
        metadata_df.at[idx, 'cover_perc'] = (g1.intersection(g2).area/g1.area)
        
def build_order_from_metadata(metadata_df, idx, row, products_bundles):
    
    sample_id = idx
    
    
    filtered_df = metadata_df[metadata_df.sample_id==sample_id]
    
    # This will create a tuple with the item_type and with the corresponding associated item_ids'
    items_by_type = [(item_type, filtered_df[filtered_df.item_type == item_type].id.to_list())
              for item_type in filtered_df.item_type.unique()]
    
    products_order = [
        {
            "item_type":k, 
            "item_ids":v, 
            "product_bundle": products_bundles[k]
        } for k, v in items_by_type
    ]
    
    # clip to AOI
    aoi_geojson = json.loads(dumps(row.geometry))
    tools = [{
        'clip': {
            'aoi': aoi_geojson
        }
    },]
    
    order_request = {
        'name': f'sample_{str(sample_id)}',
        'products': products_order,
        'tools': tools,
        'delivery': {
            'single_archive': True,
            'archive_filename':'{{name}}_{{order_id}}.zip',
            'archive_type':'zip'
        },
            'notifications': {
                       'email': False
        },
    }
    return order_request
    
    
def score_items(dataframe, item_type_score, months_score, cloud_score, cover_score):
    """Filter and score each item according to the season and item_type
    
    Return:
        Scored items dataframe.
        
    """
    # Create a copy to avoid mutate the initial df
    df = dataframe.copy()
    
    item_count_per_year = dict(df.groupby(df.date.dt.year).size())
    
    for k_year in item_count_per_year.keys():
        
        # Filter only years with more than one image
        if item_count_per_year[k_year] > 1:
            for idx, row in df.iterrows():
                
                month = row.date.month

                df.at[idx, 'season_score'] = months_score[month]
                df.at[idx, 'item_score'] = item_type_score[row['item_type']]
                df.at[idx, 'cloud_score'] = cloud_score(row['cloud_cover'])
                df.at[idx, 'covered_area'] = cover_score(row['cover_perc'])
    
    df['total_score'] = df.season_score + \
                        df.item_score + \
                        df.cloud_score + \
                        df.covered_area
    
    df = df.sort_values(by=['total_score', 'date'], ascending=False)

    return df
    
    
def get_one_item_per_year(scored_items_df):
    
    df = scored_items_df.copy()
    df['year'] = df.date.dt.year
    df = df.drop_duplicates(subset=['year'], keep='first')
    df = df.sort_values(by=['date'], ascending=False)
    
    return df
    
def track_order(order_id, client, num_loops=50):
    count = 0
    while(count < num_loops):
        count += 1
        order_info = client.get_individual_order(order_id).get()
        state = order_info['state']
        print(state)
        success_states = ['success', 'partial']
        if state == 'failed':
            raise Exception(response)
        elif state in success_states:
            break
        
        time.sleep(10)
        
def get_existing_orders(client, pages=None):
    # Search all the requested orders per page
    # Fixed api.models NEXT_KEY parameter from "_next" to "next"

    ordered_orders = client.get_orders()
    ordered_orders.NEXT_KEY = "next"
    order_pages=[]

    # We can limit the search to certain number of pages
    # if we leave as none, will search over all of them
    limit_to_x_pages = pages
    for page in ordered_orders.iter(limit_to_x_pages):
        page.NEXT_KEY = "next"
        order_pages.append(page.get())

    current_server_orders = [order for page in order_pages for order in page['orders']]
    
    return current_server_orders



def get_orders_status(client, pages=None):
    current_server_orders = get_existing_orders(client, pages)
    progress_df = pd.DataFrame([(f['created_on'], 
                                 f['last_message'], 
                                 f['last_modified'], 
                                 f['id'], 
                                 f['name'], 
                                 f['state'], ) for f in current_server_orders])
    progress_df.columns =['created_on', 'last_message', 'last_modified', 'id', 'name', 'state',]
    progress_df.sort_values(by=['created_on'])
    
    return progress_df