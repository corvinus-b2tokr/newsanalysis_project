def remove_duplicates(file_path):
    with open(file_path, 'r') as file:
        urls = file.read().splitlines()

    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)

    with open(file_path, 'w') as file:
        file.write('\n'.join(unique_urls))

remove_duplicates('../data/article_urls.txt')