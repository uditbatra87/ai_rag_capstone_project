import asyncio

async def send_data_to_db():
    print("Sending data to Database.......")
    await asyncio.sleep(20)
    print("Data Sent Successfully to Database.......")

async def my_workflow():
    print("Processing Data.......")
    task = asyncio.create_task(send_data_to_db())
    print("Moving Ahead.......")
    await asyncio.sleep(3)
    print("User Logged Out Successfully.......")
    await task   # <-- waits here until send_data_to_db() actually finishes

asyncio.run(my_workflow())
