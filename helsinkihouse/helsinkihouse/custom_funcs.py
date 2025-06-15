import itertools
import re
from random import randint
from time import sleep  # to slow down scrapping and not overload website

import pandas as pd
from bs4 import BeautifulSoup
from requests import get  # make queries look like they are from actual browser


def parse_etuovi():
    """
    Method to parse through all the entries in "https://www.etuovi.com" with
    respect to the cities of Helsinki, Espoo and Vantaa
    Inputs
    ------
        -
    """
    N_TOTAL = 200  # total number of pages to look for ads
    current = 0  # number of current page

    # Initialize empty lists for variables to store
    links = []
    titles = []
    addresses = []
    prices = []
    sizes = []
    years = []

    for page in range(N_TOTAL):
        web = "https://www.etuovi.com/myytavat-asunnot?haku=M1606820702&sivu=" + str(
            page
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
        }
        try:
            response = get(web, headers)
        except:
            print("Invalid web request.")

        # Create HTML soup
        html_soup = BeautifulSoup(response.text, "html.parser")
        house_containers = html_soup.find_all(
            "div", class_="ListPage__cardContainer__39dKQ"
        )

        if house_containers != []:
            # Iterate through every container
            for house in house_containers:
                try:
                    # Extract data
                    link = (
                        "https://www.etuovi.com"
                        + house.find("a", class_="styles__cardLink__2Oh5I")["href"]
                    )
                    title = house.find_all("h5")[0].text
                    address = house.find_all("h4")[0].text
                    price_str = house.find(string=re.compile(".*€")).replace("\xa0", "")
                    price = float(re.sub(r"\s+|€", "", price_str))
                    size_html = house.find(
                        "div",
                        class_="flexboxgrid__col-xs__26GXk flexboxgrid__col-md-4__2DYW-",
                    )
                    size_str = size_html.find_all("span")[1].text.replace(",", ".")
                    size = float(re.sub(r".m²", "", size_str))
                    year_html = house.find(
                        "div",
                        class_="flexboxgrid__col-xs-3__3Kf8r flexboxgrid__col-md-4__2DYW-",
                    )
                    year = int(year_html.find_all("span")[1].text)

                    # Append data to each list
                    links.append(link)
                    titles.append(title)
                    addresses.append(address)
                    prices.append(price)
                    sizes.append(size)
                    years.append(year)
                except:
                    print(f"Error extracting data from {house}.")

        # Wait 1-2 seconds to avoid overloading the website
        sleep(randint(1, 2))
        print(f"Page #{page} scrapped.")

        # Update page counter
        page += 1

    # Store information in a DataFrame
    df = pd.DataFrame(
        {
            "Link": links,
            "Title": titles,
            "Address": addresses,
            "Price": prices,
            "Size": sizes,
            "Year": years,
        }
    )
    return df
