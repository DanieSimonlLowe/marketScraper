from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
from urllib.parse import quote
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from website import Website
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_pages(driver):
    driver.get("https://www.marketindex.com.au/asx-listed-companies")  
    
    rows = WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "tr"))
    )

    suffix_map = {'B': 10**9, 'M': 10**6, 'K': 10**3}  # Add more as needed
    
    data = []
    for line in tqdm(rows):
        try:
            name = line.find_element(By.CLASS_NAME, "sticky-column").text
            money = line.find_element(By.CLASS_NAME, "text-right").text
            money = money.replace("$", "")
            
            money, suffix = float(money[:-1]), money[-1].upper()
            money = money * suffix_map.get(suffix, 1)  
            data.append([name,money])
        except Exception as e:  # Handle the case where the element isn't found
            continue
    return data

def get_company_website(driver,search_query):
    encoded_query = quote(search_query)
    driver.get(f"https://duckduckgo.com/?t=ffab&q={encoded_query}&ia=web")
    element = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "eVNpHGjtxRBq_gLOfGDr"))
    )

    website = element.get_attribute('href')
    return website

class ScrapeThread(threading.Thread): 
    def __init__(self,name,money,model):
        threading.Thread.__init__(self) 
        self.name = name
        self.money = money
        self.model = model
  
    def run(self): 
        driver = get_driver()
        try:
            url = get_company_website(driver,self.name)
            if url is None or 'google' in url or 'linkedin' in url or 'facebook' in url or 'marketindex' in url:
                driver.quit()
                return
            if Path(f"data/{self.name}.pkl").is_file():
                driver.quit()
                return
            site = Website(url,self.money,driver,self.model)
            driver.quit()
            with open(f"data/{self.name}.pkl", "wb") as file:
                pickle.dump(site, file)
            print(self.name + ' done')
        except Exception as e:
            print(f"An error occurred: {e}")
            driver.quit()
        

max_count = 4
def get_driver():
    service = Service(executable_path="C:\Program Files\chromedriver-win32\chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    return driver


def main():
    driver = get_driver()
    
    data = get_pages(driver)

    driver.quit()

    model = SentenceTransformer('all-MiniLM-L6-v2-32dim')
    
    threads = []
    for (name, money) in reversed(data):
        while len(threads) >= max_count:
            for t in threads:
                t.join(timeout=0.5)
                if not t.is_alive():
                    threads.remove(t)
                    break
        
        t = ScrapeThread(name,money,model)
        t.start() 
        threads.append(t)
        
         

    
    
    
    

main()
# model = SentenceTransformer('all-MiniLM-L6-v2-32dim')
# ScrapeThread('Whitefield Industrials Ltd',23.79 * 1000,model).start()
