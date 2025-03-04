from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time


# Initialize the WebDriver
driver = webdriver.Chrome()

# Function to fetch article data
def fetch_article_data(url):
    driver.get(url)
    time.sleep(3)  # Wait for the page to load

    accept_button = driver.find_element(By.XPATH, "//button[@mode='primary']")
    accept_button.click()
    time.sleep(2)  # Wait for the action to complete

    
    title_section = driver.find_element(By.CLASS_NAME, 'title-section')
    title = title_section.find_element(By.TAG_NAME, 'h1').text

    author = driver.find_element(By.CLASS_NAME, 'author__name').text if driver.find_elements(By.CLASS_NAME, 'author__name') else 'Unknown'

    facebook_section = driver.find_element(By.CSS_SELECTOR, 'div.options.options-top.spacing-top')    
    facebook_activity = facebook_section.find_element(By.TAG_NAME, 'p').text if facebook_section else '0'

    tags = [tag.text for tag in driver.find_elements(By.CSS_SELECTOR, 'a.tag.tag--basic')]
    
    article_section = driver.find_element(By.CLASS_NAME, 'article-html-content')
    article_text = ' '.join([p.text for p in article_section.find_elements(By.TAG_NAME, 'p')])
    
    return {
        'title': title,
        'author': author,
        'tags': tags,
        'facebook_activity': facebook_activity,
        'article_text': article_text
    }

# Example URL
url = 'https://telex.hu/belfold/2025/03/04/oktatas-iskolak-gerincvedo-szek-valtozas-vasarlas-julius-1'

# Fetch and store the article data
article_data = fetch_article_data(url)
print(article_data)


# Close the WebDriver
driver.quit()