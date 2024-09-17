# source/source_ski_resorts

'''
Module for gathering ski resort data.

The following data sources are used in the module:
    - skiresort.info
    - epicpass.com
    - ikonpass.com
    - united states major regions: (https://www.mappr.co/political-maps/us-regions-map/#:~:text=The%20United%20States%20of%20America%20is%20a%20country%20made%20up#:~:text=The%20United%20States%20of%20America%20is%20a%20country%20made%20up)
    - canada regions major regions: https://www.worldatlas.com/articles/the-regions-of-canada.html#:~:text=Canada%20is%20made%20up%20of%20five%20geographic%20regions,%20the%20Atlantic#:~:text=Canada%20is%20made%20up%20of%20five%20geographic%20regions,%20the%20Atlantic
'''

# import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from selenium import webdriver


'''
main data - skiresort.info
'''

# function to get basic information
def scrape_basics(page_soup, resort_info):
    # isolate to list of resorts
    resort_list = page_soup.find(id='resortList')
    
    # break down list of resorts
    resorts = resort_list.find_all(class_='panel-body middle-padding')
    
    # get basic information for each resort
    for resort in resorts:
        # link
        link = resort.find(class_='h3').find(class_='h3')['href']
        
        # location info
        name = resort.find(class_='h3').find(class_='h3').text.strip()
        print(name) # debugging
        location_list = resort.find(class_='sub-breadcrumb').find(class_='sub-breadcrumb').find_all('a')
        location_list_text = [item.text for item in location_list]
        
        # table info
        info_table = resort.find('table').find_all('tr')
        
        # rating
        try:
            rating = info_table[0].find(class_='rating-list js-star-ranking stars-middle')['data-rank']
        except:
            print(f'{name} does not contain eligible rating data')
            return resort_info
        
        # elevation
        elevation = info_table[1].text.strip()
        elevation_list = re.findall(r'\d+', elevation)
        
        # trails
        trails = info_table[2]
        # if trails aren't listed, not a "resort"
        try:
            trails_list = [float(trail) for trail in trails.text.split() if trail != 'km']
        except:
            print(f'{name} does not contain eligible trails data')
            return resort_info
        
        # lifts
        lifts = info_table[3].text.split()[0]
        
        # price
        # some resorts do not have price information
        try:
            price = re.search(r'\d+', info_table[4].text).group()
        except:
            price = None
        
        # add to data structure
        resort_info['Resort'].append(name)
        resort_info['Region'].append(location_list_text[0])
        resort_info['Country'].append(location_list_text[1])
        resort_info['Locale 1'].append(location_list_text[2])
        # some resorts have a fourth subdivision of location
        try:
            resort_info['Locale 2'].append(location_list_text[3])
        except:
            resort_info['Locale 2'].append(None)
        resort_info['Overall Rating'].append(rating)
        resort_info['Elevation Difference'].append(elevation_list[0])
        resort_info['Elevation Low'].append(elevation_list[1])
        resort_info['Elevation High'].append(elevation_list[2])
        resort_info['Trails Total'].append(trails_list[0])
        resort_info['Trails Easy'].append(trails_list[1])
        resort_info['Trails Intermediate'].append(trails_list[2])
        resort_info['Trails Difficult'].append(trails_list[3])
        resort_info['Lifts'].append(lifts)
        resort_info['Price'].append(price)
        resort_info['Link'].append(link)
        
    return resort_info

# function to scrape arrival location
def scrape_location(link, resort_info):
    # url for arriving by car
    arrival_link = f'{link}/arrival-car'
    # print for debugging and progress
    print(f'Location: {arrival_link}')
    # soup
    arrival_page = requests.get(arrival_link)
    arrival_soup = BeautifulSoup(arrival_page.text, 'lxml')
    # location information
    location = arrival_soup.find(class_='news-element').find_all('li')[1].text
    # update data storage
    resort_info['Address'].append(location)
    
    return resort_info

# function to scrape rating by criteria
def scrape_ratings(link, resort_info):
    # url for ratings
    ratings_link = f'{link}/test-report'
    # print for debugging and progress
    print(f'Ratings: {ratings_link}')
    # soup
    ratings_page = requests.get(ratings_link)
    ratings_soup = BeautifulSoup(ratings_page.text, 'lxml')
    # ratings section
    ratings_lists = ratings_soup.find_all(class_='stars-link-list')
    if ratings_lists is not None:
        # create data storage for criteria and ratings
        criteria_storage = []
        ratings_storage = []
        # loop through the ratings sections
        for ratings_list in ratings_lists:
            # get criteria from ratings section
            criteria = [crit.text.strip() for crit in ratings_list.find_all(class_='stars-link-element')]
            # get ratings from ratings section
            ratings = [rating['data-rank'] for rating in ratings_list.find_all(class_='rating-list js-star-ranking stars-middle')]
            # update data storage with values
            criteria_storage = criteria_storage + criteria
            ratings_storage = ratings_storage + ratings
        # update resort_info with given ratings
        for num, crit in enumerate(criteria_storage):
            resort_info[crit].append(ratings_storage[num])
        # update remainder of missing values from resort_info
        # find ratings which have less values than the others
        required_length = len(resort_info[criteria_storage[0]])
        for info in resort_info:
            if len(resort_info[info]) < required_length:
                resort_info[info].append(None)
        
        return resort_info

# function to create data storage for resort
def create_resort_storage():
    # keys
    resort_keys = ['Resort', 'Region', 'Country', 'Locale 1', 'Locale 2', 'Overall Rating',
                   'Elevation Difference', 'Elevation Low', 'Elevation High',
                   'Trails Total', 'Trails Easy', 'Trails Intermediate', 'Trails Difficult',
                   'Lifts', 'Price', 'Link', 'Address', 'Ski resort size', 'Slope offering, variety of runs',
                   'Lifts and cable cars', 'Snow reliability', 'Slope preparation',
                   'Access, on-site parking', 'Orientation (trail map, information boards, sign-postings)',
                   'Cleanliness and hygiene', 'Environmentally friendly ski operation',
                   'Friendliness of staff', 'Mountain restaurants, ski huts, gastronomy',
                   'Après-ski', 'Accommodation offering directly at the slopes and lifts',
                   'Families and children', 'Beginners', 'Advanced skiers, freeriders',
                   'Snow parks', 'Cross-country skiing and trails']
    
    # create dictionary
    resort_info = {key:[] for key in resort_keys}
    
    return resort_info

# culminating function for scraping ski resort data
def scrape_ski_resorts(usa_pages=4, canada_pages=3):
    # url mapping
    url = 'https://www.skiresort.info/ski-resorts'
    url_usa = f'{url}/usa'
    url_canada = f'{url}/canada'
    
    # resort data storage
    resort_info = create_resort_storage()
    
    # basics - usa pages
    for usa_page in range(usa_pages):
        if usa_page == 0:
            page = requests.get(url_usa)
            soup = BeautifulSoup(page.text, 'lxml')
            resort_info = scrape_basics(page_soup=soup, resort_info=resort_info)
            # add sleep time to space out requests
            time.sleep(1)
        else:
            page = requests.get(f'{url_usa}/page/{usa_page+1}')
            soup = BeautifulSoup(page.text, 'lxml')
            resort_info = scrape_basics(page_soup=soup, resort_info=resort_info)
            # add sleep time to space out requests
            time.sleep(1)
            
    # basics - canada pages
    for canada_page in range(canada_pages):
        if canada_page == 0:
            page = requests.get(url_canada)
            soup = BeautifulSoup(page.text, 'lxml')
            resort_info = scrape_basics(page_soup=soup, resort_info=resort_info)
            # add sleep time to space out requests
            time.sleep(1)
        else:
            page = requests.get(f'{url_canada}/page/{canada_page+1}')
            soup = BeautifulSoup(page.text, 'lxml')
            resort_info = scrape_basics(page_soup=soup, resort_info=resort_info)
            # add sleep time to space out requests
            time.sleep(1)
            
    # get links from both usa and canada
    links = resort_info['Link'].copy()
    
    # get locations and ratings
    for link in links:
        resort_info = scrape_location(link, resort_info)
        # add sleep time to space out requests
        time.sleep(1)
        resort_info = scrape_ratings(link, resort_info)
        # add sleep time to space out requests
        time.sleep(1)
    
    return resort_info

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run scraper    
resort_info = scrape_ski_resorts()

# turn dictionary into dataframe
resort_df = pd.DataFrame(resort_info)

# save initial dataframe
resort_df.to_csv('../../data/ski_resorts_data.csv', index=False)
'''


'''
passes - epicpass.com
'''

# function to scrape epic resorts data
def scrape_epic():
    # base url
    url = 'https://www.epicpass.com/'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'lxml')
    
    resorts_list = soup.find_all(class_='footer__panel__list--resorts')
    epic_info = {'Area':[], 'Resort': []}
    for section in resorts_list:
        area = section.find(class_='footer__panel__heading--resorts').text.strip()
        area_resorts = section.find_all(class_='footerlink')
        for resort in area_resorts:
            epic_info['Area'].append(area)
            epic_info['Resort'].append(resort.text.split(', ')[0].strip())
    
    # turn into dataframe
    epic_df = pd.DataFrame(epic_info)
    
    return epic_df

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run function
epic_df = scrape_epic()

# save dataframe
epic_df.to_csv('../../data/epic_resorts.csv', index=False)
'''


'''
passes - ikonpass.com
'''

# function to scrape ikon resorts data
def scrape_ikon():
    # base url
    url = 'https://www.ikonpass.com/en/destinations'
    
    # has javascript component, initiate with selenium
    driver = webdriver.Chrome()
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    # resorts section
    resorts_list = soup.find(class_='destination-nav container')
    country_split = resorts_list.find_all('h2')
    countries = [country.text.lower().replace(' ', '-') for country in country_split]
    country_sections = []
    for country in countries:
        country_section = resorts_list.find(class_=f'destination-nav__country destination-nav__country--{country}')
        country_sections.append(country_section)
        
    # extract required information
    ikon_info = {'Country': [], 'Region': [], 'Resort': []}
    for num, country in enumerate(country_sections):
        regions_sections = country.find(class_='region-items').find_all('div')
        for region_section in regions_sections:
            if region_section.find('h3') is not None:
                region = region_section.find('h3').text
                resorts_visible = region_section.find_all(class_='mobile-visible')
                resorts_hidden = region_section.find_all(class_='mobile-hidden')
                resorts = resorts_visible + resorts_hidden
                for resort in resorts:    
                    # append results
                    ikon_info['Country'].append(countries[num])
                    ikon_info['Region'].append(region)
                    ikon_info['Resort'].append(resort.text)
                    
    # turn info into dataframe
    ikon_df = pd.DataFrame(ikon_info)
    
    return ikon_df

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run ikon scraper
ikon_df = scrape_ikon()

# save dataframe
ikon_df.to_csv('../../data/ikon_resorts.csv', index=False)
'''


'''
major regions - united states
'''

# function to scrape united states regions
def scrape_regions_us():
    url = 'https://www.mappr.co/political-maps/us-regions-map/#:~:text=The%20United%20States%20of%20America%20is%20a%20country%20made%20up#:~:text=The%20United%20States%20of%20America%20is%20a%20country%20made%20up'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    # table
    table = soup.find('table')
    table_header = [header.text for header in table.find_all('th')]
    table_data = {key:[] for key in table_header}
    table_body = table.find('tbody')
    for row in table_body.find_all('tr'):
        row_data = row.find_all('td')
        for num, col in enumerate(row_data):
            table_data[table_header[num]].append(col.text)
    
    # dataframe
    df = pd.DataFrame(table_data)
    return df

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run function
us_regions = scrape_regions_us()
us_regions.to_csv('../../data/us_regions.csv', index=False)
'''


'''
major regions - canada
'''

# function to scrape canada regions
def scrape_regions_canada():
    url = 'https://www.worldatlas.com/articles/the-regions-of-canada.html#:~:text=Canada%20is%20made%20up%20of%20five%20geographic%20regions,%20the%20Atlantic#:~:text=Canada%20is%20made%20up%20of%20five%20geographic%20regions,%20the%20Atlantic'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    # table
    table = soup.find('table')
    table_header = [header.text for header in table.find_all('th')]
    table_data = {key:[] for key in table_header}
    table_body = table.find('tbody')
    for row in table_body.find_all('tr'):
        row_data = row.find_all('td')
        for num, col in enumerate(row_data):
            table_data[table_header[num]].append(col.text)
            
    # dataframe
    df = pd.DataFrame(table_data)
    return df

'''
# BLOCK COMMENTED OUT AFTER RUNNING TO REDUCE CALLS
# run function
canada_regions = scrape_regions_canada()
canada_regions.to_csv('../../data/canada_regions_raw.csv', index=False)
'''
