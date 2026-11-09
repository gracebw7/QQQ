from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


PgInteger = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]
Location = Annotated[PgInteger, Field(ge=1)]
Embedding = Annotated[list[FiniteFloat], Field(min_length=384, max_length=384)]


class DatabaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class DocumentCreate(DatabaseModel):
    external_id: str
    title: str | None = None
    source: str | None = None
    content: str


class Document(DocumentCreate):
    id: PgInteger


class LineRange(DatabaseModel):
    start_line: Location
    end_line: Location

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ChunkCreate(LineRange):
    document_id: PgInteger
    content: str
    embedding: Embedding


class Chunk(ChunkCreate):
    id: PgInteger


class PageLineRange(LineRange):
    """Page mapping supplied before a document has a database ID."""
    page_num: Location


class PageLineCreate(PageLineRange):
    document_id: PgInteger


class PageLine(PageLineCreate):
    id: PgInteger

# OLD DATA MODEL
# 
# 
# from typing import Annotated, Self
# 
# from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
# 
# 
# PgInteger = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]
# Location = Annotated[PgInteger, Field(ge=1)]
# Embedding = Annotated[list[FiniteFloat], Field(min_length=384, max_length=384)]
# 
# 
# class DatabaseModel(BaseModel):
#     model_config = ConfigDict(extra="forbid", from_attributes=True)
# 
# 
# class DocumentCreate(DatabaseModel):
#     external_id: str
#     title: str | None = None
#     source: str | None = None
#     content: str
# 
# 
# class Document(DocumentCreate):
#     id: PgInteger
# 
# 
# class HeadingCreate(DatabaseModel):
#     document_id: PgInteger
#     parent_id: PgInteger | None = None
#     level: Annotated[int, Field(ge=1, le=6)]
#     title: str
#     position: PgInteger
#     start_line: Location
#     end_line: Location
# 
#     @model_validator(mode="after")
#     def validate_line_range(self) -> Self:
#         if self.end_line < self.start_line:
#             raise ValueError("end_line must be greater than or equal to start_line")
#         return self
# 
# 
# class Heading(HeadingCreate):
#     id: PgInteger
# 
# 
# class ChunkCreate(DatabaseModel):
#     document_id: PgInteger
#     heading_id: PgInteger | None = None
#     position: PgInteger
#     paragraph_number: Location
#     content: str
#     embedding: Embedding
# 
# 
# class Chunk(ChunkCreate):
#     id: PgInteger
# 
