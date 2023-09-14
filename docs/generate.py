import os

from .generator import DocsGenerator

MD_PATH = os.getenv('MD_PATH', 'DOCS.md')


def main():
    docs = DocsGenerator().run()
    with open(MD_PATH, 'w') as f:
        f.write(docs)


if __name__ == '__main__':
    main()
