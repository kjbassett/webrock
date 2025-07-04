import asyncio
from app import create_app


def main():
    app = asyncio.run(create_app())
    app.run(host="0.0.0.0", port=8000, single_process=True)


if __name__ == "__main__":
    main()
