import pandas as pd
import asyncio
import aiohttp
import os
import os
from multiprocessing import Pool

from tqdm import tqdm
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image
# # скачиваем фото

from tqdm import tqdm


desired_cpu_cores = "1-42"  # Указываем диапазон CPU-ядер

pid = os.getpid()  # Получаем идентификатор текущего процесса
os.system(f"taskset -p -c {desired_cpu_cores} {pid}")  # Привязываем процесс к указанным ядрам


def check_image_existence(img_path: str) -> bool:
    if os.path.isfile(img_path):
        try:
            img = Image.open(img_path)
            img.close()
            return True
        except:
            return False
    else:
        return False

class ClientBasket:
    def __init__(self, folder):
        self.folder = folder

    async def request(self, session, photo_id):
        retry_count = 10
        while retry_count >= 0:
            try:
                async with session.get(photo_id, timeout=5) as res:
                    if res.status == 200:
                        photo_name = photo_id.split("/")[-1]
                        photo_path = os.path.join(self.folder, photo_name)
                        with open(photo_path, 'wb') as f:
                            f.write(await res.read())
                        return 1
                    return b''
            except Exception as e:
                retry_count -= 1
        return b''

async def load_photos(photo_ids: list[str], data_folder: str) -> None:
    basket = ClientBasket(data_folder)
    async with aiohttp.ClientSession() as session:
        tasks = [basket.request(session, photo_id) for photo_id in photo_ids]
        await tqdm_asyncio.gather(*tasks, desc="Downloading Data from Basket", leave=False)


def load_photos_from_df_batched(df: pd.DataFrame, data_folder: str) -> pd.DataFrame:
    batch_size = 4000
    # Create batches of IDs
    df_batches = [df.iloc[i:i+batch_size].copy() for i in range(0, len(df.photo_url.tolist()), batch_size)]
    result_dfs = []
    logger.info(f"Split db_data into {len(df_batches)} batches")

    for idx, requested_batch_df in enumerate(df_batches):
        logger.info(f"Loading data for batch {idx} of {len(df_batches)}")

        asyncio.run(load_photos(requested_batch_df.photo_url.tolist(), data_folder))

        logger.info("Data was successfully loaded")

        with Pool(processes=os.cpu_count()) as pool:
            poll_data = [os.path.join(data_folder, photo_url.split("/")[-1])
                         for photo_url in requested_batch_df.photo_url.tolist()]
            check_results = list(pool.map(check_image_existence,poll_data))

        requested_batch_df["img_path"] = poll_data
        batch_df = requested_batch_df.loc[check_results]
        result_dfs.append(batch_df)

        remaining_count = len(requested_batch_df) - len(batch_df)
        logger.info(f"Lost images count: {remaining_count}")

    result_df = pd.concat(result_dfs, ignore_index=True)
    return result_df



if __name__ == "__main__":

    df_path = "dataframes/parsed_toloka_dataset.csv" 
    df = pd.read_csv(df_path)
    data_folder = os.path.abspath("toloka_parsed_data")
    os.makedirs(data_folder, exist_ok=True)
    result_df = load_photos_from_df_batched(df, data_folder)
    result_df.to_csv("dataframes/loaded_parsed_toloka_dataset.csv", index=False)
