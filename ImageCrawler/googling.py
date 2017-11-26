# -*- coding: utf-8 -*-

import logging
from datetime import date

from icrawler.builtin import (GoogleImageCrawler)


def googling():
    google_crawler = GoogleImageCrawler(
        downloader_threads=4,
        storage={'root_dir': '/path'},
        log_level=logging.INFO)
    google_crawler.crawl(
        'keywords',
        max_num=1000,
        date_min=date(2000, 2, 1),
        date_max=date(2017, 10, 30))

if __name__ == '__main__':
    googling()
