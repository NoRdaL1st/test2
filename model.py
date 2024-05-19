
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, RequestInput, ResponseOutput
from mlserver.codecs import StringCodec

class CustomFinancialSentimentModel(MLModel):
    async def load(self):
        # Загрузка модели и токенайзера из Hugging Face
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.ready = True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        # Извлечение текстовых данных из запроса
        inputs = [inp.data for inp in payload.inputs if inp.name == "text"]
        texts = StringCodec.decode(inputs[0])

        # Токенизация текста
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        # Предсказание
        with torch.no_grad():
            outputs = self.model(**tokens)

        # Получение предсказаний
        predictions = outputs.logits.argmax(dim=-1).tolist()

        # Формирование ответа
        response = InferenceResponse(
            model_name=self.name,
            outputs=[ResponseOutput(
                name="predictions",
                shape=[len(predictions)],
                datatype="INT64",
                data=predictions
            )]
        )
        return response
