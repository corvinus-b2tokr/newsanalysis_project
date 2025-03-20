from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import json

# Function to get article URLs from a page
def get_article_urls(page_number, driver):
    driver.get(f"https://telex.hu/legfrissebb?oldal={page_number}")

    info_divs = driver.find_elements(By.CLASS_NAME, 'list__item__info')

    articles = []

    for info_div in info_divs:
        article_link = info_div.find_element(By.TAG_NAME, 'a')
        date = info_div.find_element(By.CLASS_NAME, 'article_date').find_element(By.TAG_NAME, 'span').text

        if 'január' in date:
            articles.append(article_link.get_attribute('href'))

    return articles


# Function to get all article URLs
def get_all_articles(driver):
    page_number = 330
    all_articles = []

    while True:
        articles = get_article_urls(page_number, driver)
        all_articles.extend(articles)
        page_number += 1

        if page_number==529:
            break

    with open('../data/article_urls_4.txt', 'w') as file:
        file.write('\n'.join(all_articles))


# Function to scrape article data
def scrape_article_data(url, driver):
    driver.get(url)

    try:
        accept_button = driver.find_element(By.XPATH, "//button[@mode='primary']")
        accept_button.click()
    except NoSuchElementException:
        pass


    title_section = driver.find_element(By.CLASS_NAME, 'title-section')
    title = title_section.find_element(By.TAG_NAME, 'h1').text

    author = driver.find_element(By.CLASS_NAME, 'author__name').text if driver.find_elements(By.CLASS_NAME, 'author__name') else 'Unknown'

    try:
        facebook_section = driver.find_element(By.CSS_SELECTOR, 'div.options.options-top.spacing-top')
        facebook_activity = facebook_section.find_element(By.TAG_NAME, 'p').text
    except NoSuchElementException:
        return None

    tags = [tag.text for tag in driver.find_elements(By.CSS_SELECTOR, 'a.tag.tag--basic')]

    lead_text = ''
    try:
        lead_paragraph = driver.find_element(By.CSS_SELECTOR, 'p.article-html-content.article__lead')
        lead_text = lead_paragraph.text
    except NoSuchElementException:
        pass

    article_text = ''
    try:
        article_section = driver.find_element(By.CSS_SELECTOR, 'div.article-html-content')
        article_text = ' '.join([p.text for p in article_section.find_elements(By.TAG_NAME, 'p')])
    except NoSuchElementException:
        pass

    full_article_text = f"{lead_text} {article_text}".strip()

    return {
        'title': title,
        'author': author,
        'tags': tags,
        'facebook_activity': facebook_activity,
        'article_text': full_article_text,
        'url': url
    }

def store_article_data(driver):
    with open('../data/article_urls.txt', 'r') as file:
        article_urls = file.read().splitlines()
    for i in range(850, len(article_urls), 5):
        article_data = []
        batch_urls = article_urls[i:i+5]


        for url in batch_urls:
            print(f'Scraping article {article_urls.index(url) + 1}/{len(article_urls)} {url}')
            data = scrape_article_data(url, driver)
            if data:
                article_data.append(data)



        with open(f'../data/article_data_{i+1}-{i+len(batch_urls)}.json', 'w', encoding='utf-8') as file:
            json.dump(article_data, file, indent=2, ensure_ascii=False)

        print(f"Stored {len(article_data)} articles in '../data/article_data_{i+1}-{i+len(batch_urls)}.json'")



def main():
    driver = webdriver.Chrome()
    #get_all_articles(driver)
    store_article_data(driver)
    driver.quit()

if __name__ == '__main__':
    main()