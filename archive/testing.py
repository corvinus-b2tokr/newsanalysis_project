from selenium import webdriver
from selenium.webdriver.common.by import By

# URL of the page to scrape
url = "https://telex.hu/legfrissebb?oldal=30"

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the URL
driver.get(url)

# Find all divs with the class 'article_date'
article_date_divs = driver.find_elements(By.CLASS_NAME, 'article_date')

# Count the number of divs with the class 'article_date' that have the number 28 inside the span element
num_article_date_divs_with_28 = sum(1 for div in article_date_divs if div.find_element(By.TAG_NAME, 'span') and ('február 28.' in div.find_element(By.TAG_NAME, 'span').text or '28. February' in div.find_element(By.TAG_NAME, 'span').text))

# Print the result
print(f"The number of divs with the class 'article_date' that have the number 28 inside the span element is: {num_article_date_divs_with_28}")

# Close the WebDriver
driver.quit()