from .imports import *

def strip_punct(token):
    from string import punctuation
    return token.strip(punctuation)


# Quick function to download a webpage
def gethtml(url,timeout=10):
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    res = session.get(url)
    url2 = res.url
    if url!=url2: res = session.get(url2)
    return res.text