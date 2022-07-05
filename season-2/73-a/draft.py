# Example of how BeautifulSoup won't work.
# HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36'}
# pages = range(1, PAGES)
# session = requests.Session()
# for page in pages:
#     url = BASE.format(page)
#     response = session.get(url, headers=HEADERS)
#     soup = BeautifulSoup(response.content)
#     break