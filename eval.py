from documents import FileSystem
import asyncio


async def main():
    loader = FileSystem("skolkovo")
    async for doc in loader:
        print(doc.text)

        # print(doc.claims)
        # print(doc.description)
        break


asyncio.run(main())
