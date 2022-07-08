from ..imports import *


# Scrape IMDB page
URL_CACHE={}
def get_cast_from_imdb(imdb_id):
    global URL_CACHE
    url=f'https://www.imdb.com/title/tt{imdb_id}/fullcredits'
    if url not in URL_CACHE:
        html=gethtml(url)
        URL_CACHE[url] = html
    else:
        html = URL_CACHE[url]


    # parse with beautiful soup
    import bs4
    dom = bs4.BeautifulSoup(html, 'lxml')

    # find the cast_list table
    table = dom.select_one('.cast_list')

    # Quickly assemble a table
    old=[]
    for row in table('tr'):
        cells = row('td')
        if len(cells)<4: continue

        actor_cell = cells[1]
        char_cell = cells[3]

        odx={}
        odx['actor_name'] = actor_cell.text.replace('\n',' ').strip()
        # odx['actor_fname'] = odx['actor_name'].split()[0]
        # odx['actor_lname'] = odx['actor_name'].split()[-1]

        actor_link = actor_cell.select_one('a')
        actor_url = actor_link.attrs['href']
        if not actor_url.startswith('/name/'): continue
        odx['actor_id'] = actor_url[len('/name/'):].replace('/','')


        odx['char_name'] = char_cell.text.replace('\n',' ').strip()
        # odx['char_fname'] = odx['char_name'].split()[0]
        # odx['char_lname'] = odx['char_name'].split()[-1]

        char_link = char_cell.select_one('a')
        odx['char_id'] = ''
        if char_link: char_url = char_link.attrs['href']
        if '/characters/' in char_url: odx['char_id'] = char_url.split('/characters/',1)[-1].split('?')[0]

        old.append(odx)

    return pd.DataFrame(old)