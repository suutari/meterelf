import os

DEBUG = {
    x for x in os.getenv('DEBUG', '').replace(',', ' ').split()
    if x.lower() not in {'0', 'no', 'off', 'false'}
}

if 'all' in DEBUG:
    DEBUG = {'masks'}
