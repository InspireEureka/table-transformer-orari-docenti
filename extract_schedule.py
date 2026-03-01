import json
import easyocr
from transformers import TATRModel, TATRTokenizer

class OrariExtractor:
    def __init__(self, model_name='your_pretrained_model_name'):
        self.reader = easyocr.Reader(['en'])
        self.model = TATRModel.from_pretrained(model_name)
        self.tokenizer = TATRTokenizer.from_pretrained(model_name)

    def extract_text_from_image(self, image_path):
        """
        Extracts text from an image using EasyOCR.
        """
        result = self.reader.readtext(image_path)
        return ' '.join([text[1] for text in result])

    def process_text_with_tatr(self, extracted_text):
        """
        Process the extracted text using the TATR model.
        """
        inputs = self.tokenizer(extracted_text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

    def extract_schedule(self, image_path):
        """
        Main method to extract the schedule data from an image and return it in JSON format.
        """
        raw_text = self.extract_text_from_image(image_path)
        processed_data = self.process_text_with_tatr(raw_text)
        return json.dumps(processed_data, ensure_ascii=False)

# Example usage:
# extractor = OrariExtractor()
# schedule_json = extractor.extract_schedule('path_to_schedule_image.jpg')
# print(schedule_json)