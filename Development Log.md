# Fortress

## 2022.04.10
My goal today was to design exactly what I want from Fortress. In that, I wanted to
describe the api I wanted for users.

My current idea is to use dataclasses in conjunction with context managers:
```python
@Model
class User:
    _id: int # Auto-injected?
    name: str

with Session() as s:
    new_user = User("Ian Kollipara")
    s.add(new_user)
    s.commit()

with SessionTest() as s:
    new_user = User("Ian Kollipara")
    s.add(new_user)
    s.commit()

assert User.get(1).name == "Ian Kollipara"
```

I'm also considering creating a Hypothesis plugin for testing as well, since I want to
move my testing in that direction. 

## 2022.04.14
I've had an idea that might work for handling queries. My plan is to use the Python AST module to parse
queries into their base components, then build the actual query myself. This has to work with the models
as well however, and I need a graceful way to handle this. As of now I've switched from dataclasses to 
Pydantic, as it better suited what I wanted. However, I do have some ideas on how to handle it via dataclasses.
The current idea is to use a base dataclass, similar to how I'm using pydantic. 

## 2022.05.05
I finally finished to decompiler and the Query Visitor. This is by far the most complex part of the project, but it's finally done. The next steps will be integrating it within the rest of the library's functionality.

## 2022.05.14
After updating my computer, and switching editors, I've come back to this project. My biggest plan is to iron out what I want the API to look like for this project. From what I've scaffolded, I'm looking at an API built around a main Fortress Class. 
