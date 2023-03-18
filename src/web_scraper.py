from bs4 import BeautifulSoup
from requests import get
from time import sleep
from random import randint
import re
import pandas as pd

class EtuoviScraper:
    def __init__(self, cities, num_pages=200):
        self.cities = cities
        self.num_pages = num_pages

    def scrape(self):
        links = []
        titles = []
        addresses = []
        prices = []
        sizes = []
        years = []

        for city in self.cities:
            for page in range(self.num_pages):
                web = f"https://www.etuovi.com/myytavat-asunnot?haku=M1606820702&cr={city}&sivu={page}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                try:
                    response = get(web, headers=headers)
                    response.raise_for_status()
                except Exception as e:
                    print(f"Invalid web request for {city} page {page}: {e}")
                    continue

                html_soup = BeautifulSoup(response.text, 'html.parser')
                house_containers = html_soup.find_all('div', class_='ListPage__cardContainer__39dKQ')

                if not house_containers:
                    break

                for house in house_containers:
                    try:
                        link = "https://www.etuovi.com" + house.find('a', class_='styles__cardLink__2Oh5I')['href']
                        title = house.find_all('h5')[0].text
                        address = house.find_all('h4')[0].text
                        price_str = house.find(string=re.compile(".*€")).replace('\xa0', '')
                        price = float(re.sub(r"\s+|€", "", price_str))
                        size_html = house.find('div', class_="flexboxgrid__col-xs__26GXk flexboxgrid__col-md-4__2DYW-")
                        size_str = size_html.find_all('span')[1].text.replace(',', '.')
                        size = float(re.sub(r".m²", "", size_str))
                        year_html = house.find('div', class_="flexboxgrid__col-xs-3__3Kf8r flexboxgrid__col-md-4__2DYW-")
                        year = int(year_html.find_all('span')[1].text)

                        links.append(link)
                        titles.append(title)
                        addresses.append(address)
                        prices.append(price)
                        sizes.append(size)
                        years.append(year)
                    except Exception as e:
                        print(f"Error extracting data from {house}: {e}")

                sleep(randint(1, 2))
                print(f"Scraped {city} page {page + 1}")

        df = pd.DataFrame({
            "Link": links,
            "Title": titles,
            "Address": addresses,
            "Price": prices,
            "Size": sizes,
            "Year": years
        })

        return df

if __name__ == "__main__":
    scraper = EtuoviScraper(["helsinki", "espoo", "vantaa"])
    df = scraper.scrape()
    df.to_csv("data.csv", index=False)