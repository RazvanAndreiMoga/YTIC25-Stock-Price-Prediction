import time
from pytrends import dailydata

class PyTrendsFetcher:
    def __init__(self, max_retries=5, wait_time=60):
        self.max_retries = max_retries
        self.wait_time = wait_time

    def fetch_data_with_retry(self, keyword, start_year, start_mon, stop_year, stop_mon, geo=''):
        retries = 0
        while retries < self.max_retries:
            try:
                res = dailydata.get_daily_data(keyword, start_year, start_mon, stop_year, stop_mon, geo)
                return res
            except Exception as e:
                retries += 1
                print(f"Error encountered: {e}. Retrying {retries}/{self.max_retries}...")
                time.sleep(self.wait_time)
        raise Exception("Max retries exceeded. Could not fetch data.")