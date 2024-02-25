from decimal import *
from datetime import *
from typing import *
from marshmallow import Schema, fields, post_load
from openfabric_pysdk.utility import SchemaUtil


################################################################
# SimpleText concept class - AUTOGENERATED
###############################################################
class SimpleText:
    text: List[str] = []


################################################################
# SimpleTextSchema concept class - AUTOGENERATED
################################################################
class SimpleTextSchema(Schema):
    text = fields.List(fields.String())

    @post_load
    def create(self, data, **kwargs):
        simple_text_instance = SimpleText()
        simple_text_instance.text =data.get('text',[])
        return  simple_text_instance
