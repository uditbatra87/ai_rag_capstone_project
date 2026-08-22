from pydantic import BaseModel,Field
import warnings
warnings.filterwarnings("ignore")

class User(BaseModel):
    name : str = Field(min_length=3,max_length=100)
    age : int = Field(gt=0,min=1,max=125,description="Age of the person")
    occupation : str = Field(default=None)
    hobbies : list[str] = Field(default=[])
    email : str = Field(default=None)
    phone : str = Field(default=None)


try:

    user1 = User(name="Udit",age=39,occupation='Engineer',hobbies=['Reading','Sleeping'])
    print(user1)

    user2 = User(name="Ajay",age=39)
    print(user2)

except Exception as e:
    print(e)


