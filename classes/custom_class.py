
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class AnswerParser(BaseModel):
    """
    A model for returning an answer about bikes that a customer could buy.
    Attributes:
        bike (str): The name of the bike the answer is referencing.
        answer (str): The answer to the customers question.
    """
    bike: str = Field("The name of the bike the answer is referencing.")
    answer: str = Field("The answer to the customers question.")
