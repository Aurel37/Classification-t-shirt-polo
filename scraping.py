from selenium import webdriver
import time
from urllib.request import urlopen, Request
import numpy as np
import cv2
import pandas as pd


def url_open(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
    req = Request(url=url, headers=headers)
    try:
        req = urlopen(req).read()
        img = np.array(bytearray(req), dtype='uint8')
        return cv2.imdecode(img, cv2.IMREAD_COLOR)
    except Exception as e:
        print('Error :', e)
        return None


def scrapurl(keyword, first_pos, number):
    url = "https://www.google.co.in/search?q="+keyword+"&source=lnms&tbm=isch"
    driver = webdriver.Chrome('/Users/aurelienpion/Documents/cours_enpc_2A/projet_dept/classification/chromedriver')
    driver.get(url)
    for i in range(int((number + first_pos)/400 + 1)):
        for j in range(10):
            driver.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(0.2)
        time.sleep(2.5)
        try:
            driver.find_element_by_xpath("//input[@value='Afficher plus de résultats']").click()
            time.sleep(2.5)
        except Exception as e:
            print("Less images found: " + str(e))
            break
    info_img = driver.find_elements_by_class_name('rg_i')
    urls = []
    for img in info_img:
        url = img.get_attribute('src')
        urls.append(url)
    print(len(urls))
    return urls


def writef_fcsv(urls, path):
    with open(path, 'a') as f:
        for url in urls:
            try:
                f.write(url + '\n')
            except:
                pass


if __name__ == "__main__":
    path = "images/"
    
    shirt_db = pd.read_csv("jumia_chemises.csv")
    polo_db = pd.read_csv("jumia_polos.csv")

    shirt_db = shirt_db['dsc_image_url'].to_numpy()
    polo_db = polo_db['dsc_image_url'].to_numpy()
    """
    path_key = ['polo', 'shirt', 't_shirt']
    keywords = ['polo shirt', 'chemise', 't-shirt']
    set_size = [720, 740, 750]
    pos_first_img = [0, 0, 0]
    for i, key in enumerate(keywords):
        with open(path + path_key[i] + '_url.txt', 'a') as f:
            urls = scrapurl(key, pos_first_img[i], set_size[i])
            for url in urls:
                try:
                    f.write(url + '\n')
                except:
                    pass
    """
    writef_fcsv(shirt_db, 'images/shirt_url.txt')
    writef_fcsv(polo_db, 'images/polo_url.txt')
