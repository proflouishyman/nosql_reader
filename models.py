# No longer necessary since it is NoSQL db. 

# from app import db, ma

# class OCRText(db.Model):
#     __tablename__ = 'ocr_text'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, nullable=False)
#     text = db.Column(db.Text, nullable=False)

#     summary = db.relationship('Summary', back_populates='ocr_text', uselist=False)
#     named_entities = db.relationship('NamedEntity', back_populates='ocr_text')
#     dates = db.relationship('Date', back_populates='ocr_text')
#     monetary_amounts = db.relationship('MonetaryAmount', back_populates='ocr_text')
#     relationships = db.relationship('Relationship', back_populates='ocr_text')
#     document_metadata = db.relationship('DocumentMetadata', back_populates='ocr_text', uselist=False)
#     translation = db.relationship('Translation', back_populates='ocr_text', uselist=False)
#     file_info = db.relationship('FileInfo', back_populates='ocr_text', uselist=False)

# class Summary(db.Model):
#     __tablename__ = 'summary'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     text = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='summary')

# class NamedEntity(db.Model):
#     __tablename__ = 'named_entities'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     entity = db.Column(db.Text, nullable=False)
#     type = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='named_entities')

# class Date(db.Model):
#     __tablename__ = 'dates'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     date = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='dates')

# class MonetaryAmount(db.Model):
#     __tablename__ = 'monetary_amounts'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     amount = db.Column(db.Text, nullable=False)
#     category = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='monetary_amounts')

# class Relationship(db.Model):
#     __tablename__ = 'relationships'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     entity1 = db.Column(db.Text, nullable=False)
#     relationship = db.Column(db.Text, nullable=False)
#     entity2 = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='relationships')

# class DocumentMetadata(db.Model):
#     __tablename__ = 'metadata'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     document_type = db.Column(db.Text, nullable=False)
#     period = db.Column(db.Text, nullable=False)
#     context = db.Column(db.Text, nullable=False)
#     sentiment = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='document_metadata')

# class Translation(db.Model):
#     __tablename__ = 'translation'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     french_text = db.Column(db.Text, nullable=False)
#     english_translation = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='translation')

# class FileInfo(db.Model):
#     __tablename__ = 'file_info'
#     id = db.Column(db.Text, primary_key=True)
#     file = db.Column(db.Text, db.ForeignKey('ocr_text.id'), nullable=False)
#     original_filepath = db.Column(db.Text, nullable=False)

#     ocr_text = db.relationship('OCRText', back_populates='file_info')

# # Add Marshmallow schemas if needed
# class OCRTextSchema(ma.SQLAlchemyAutoSchema):
#     class Meta:
#         model = OCRText
