from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    if request.method == 'GET':
        input_text = request.GET.get('input_text', '')

        # Sentiment analysis
        roberta = "cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(roberta)
        tokenizer = AutoTokenizer.from_pretrained(roberta)
        labels = ['Negative', 'Neutral', 'Positive']

        encoded_text = tokenizer(input_text, return_tensors='pt')
        output = model(**encoded_text)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        negative_probability = scores[0] * 100

        severity = 'Not Depressed'

        if 1 <= negative_probability <= 50:
            severity = 'Not Depressed'
        elif 51 <= negative_probability <= 89:
            severity = 'Mild'
        elif 90 <= negative_probability <= 94:
            severity = 'Moderate'
        elif 95 <= negative_probability <= 100:
            severity = 'Severe'

        result_context = {
            'result': 'Negative',  # Set result to Negative since you are customizing based on the negative class
            'severity': severity
        }

        return render(request, 'predict.html', context=result_context)
