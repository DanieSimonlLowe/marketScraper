from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer, models
import pickle
import re
import time

class Website:
    def __init__(self,url, money,driver,encoder) -> None:
        
        self.graph = {}
        self.urls = [url]
        self.money = money
        self.contents = []
        self.spliters = ['\n','.',',',' ']
        self.depth = [0]

        explored = 0
        while explored < len(self.urls):
            next = self.getChildren(self.urls[explored],driver)
            if self.depth[explored] >= 10:
                explored += 1
                continue
            url_map = []
            for temp in next:
                if not temp in self.urls:
                    url_map.append(len(self.urls))
                    self.depth.append(self.depth[explored] + 1)
                    self.urls.append(temp)
                else:
                    id = self.urls.index(temp)
                    url_map.append(id)
                    self.depth[id] = min(self.depth[id], self.depth[explored] + 1)
            self.graph[explored] = url_map
            self.contents.append(self.getContents(url,driver,encoder))
            explored += 1
        
        to_remove = []
        for id in self.graph[0]:
            in_count = 0
            for key in self.graph.keys():
                if not id in self.graph[key]:
                    in_count += 1
            if in_count >= len(self.graph.keys()) * 0.8:
                to_remove.append(id)
        
        for key in self.graph.keys():
            self.graph[key] = [id for id in self.graph[key] if not id in to_remove]
        
        
        
    
    def getContents(self,url,driver,encoder):
        # succesed = False
        # for i in range(10):
        #     try:
        #         driver.get(url)
        #         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        #         succesed = True
        #         break
        #     except:
        #         time.sleep(1)
        #         continue
        # if not succesed:
        #     return []
        
        paragraphs = []
        # Get all elements and filter for those with visible text
        elements = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, "//*"))
        )
        for element in elements:
            try:
                text = element.text.strip()
                no_num = re.sub(r'\d', '', text) # to make it ignore numbers
                if len(no_num) > 5 and len(no_num.split()) > 3:  # Filter out elements with empty text
                    text = re.sub(r'\d', '1', text) # to prevent them saying there market cap
                    parts = self.splitIntoParts(text,encoder.tokenizer)
                    paragraphs += parts
            except:
                pass
        paragraphs = list(set(paragraphs))
                
        return encoder.encode(paragraphs)
    

    def splitIntoParts(self,text,tokenizer,depth=0):
        if len(tokenizer.encode(text)) <= 256:
            return [text]
        if depth >= len(self.spliters):
            return []
        spliter = self.spliters[depth]
        parts = text.split(spliter)
        out = []
        last = 0
        for i in range(1,len(parts)):
            temp = spliter.join(parts[last:i+1])
            token_count = len(tokenizer.encode(temp))
            if token_count > 256:
                temp2 = spliter.join(parts[last:i])
                if len(tokenizer.encode(temp2)) > 256:
                    out += self.splitIntoParts(temp2,tokenizer,depth+1)
                else:
                    out.append(temp2)
                last = i
        return out
        

    def getChildren(self,url,driver):
        succesed = False
        for i in range(5):
            try:
                driver.get(url)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                succesed = True
                break
            except:
                time.sleep(0.5)
                continue
        if not succesed:
            return []

        time.sleep(0.5)
        try:
            links = WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
            )
        except:
            return []

        parsed_url = urlparse(url)
        base = f"{parsed_url.scheme}://{parsed_url.netloc}"
        temp = []
        for link in links:
            try:
                temp.append(link.get_attribute('href'))
            except:
                continue
        links = [link for link in temp if link is not None and not '.pdf' in link and not '.mp3' in link and not '.png' in link and not '.jpg' in link]
        links = [link.split('#')[0] for link in links]
        links = [link.split('?')[0] for link in links]
        links = [link for link in links if base in link]
        
        out = set()
        for link in links:
            if link[0] == '/':
                out.add(base+link)
            else:
                out.add(link)
        return out
    
    def save(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Load a Website object from a file
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def main():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Run in headless mode (no browser window)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    model = SentenceTransformer('all-MiniLM-L6-v2-32dim')

    website = Website('https://www.asb.co.nz/',23,driver,model)
    print(website.graph)
    driver.quit()


#main()