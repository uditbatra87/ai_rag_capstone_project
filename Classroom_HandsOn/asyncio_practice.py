import asyncio


#Example 1:
async def test():
    print("Test")


async def hello():
    print("Hello")
    await test()

asyncio.run(hello())

#Example 2:
async def test():
    print("Test")


async def hello():
    print("Hello")

async def main():
   await asyncio.gather(hello(),test())
    

asyncio.run(main())


