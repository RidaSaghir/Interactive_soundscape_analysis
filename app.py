
import logging

logging.basicConfig(level=logging.DEBUG)

from lts.frontend import FrontEndLite

app = FrontEndLite()

if __name__ == "__main__":
    app.launch()

