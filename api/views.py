from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np

from django.template import loader, RequestContext
#from django.http import HttpResponseRedirect

# Create your views here.

@api_view(['GET'])
def index(request):
    #return_data = { "error" : "0",  "message" : "Successful" }
    template = loader.get_template('api/index.html')
    context = {}
    #return Response(template.render(context))
    return render(request, 'api/index.html', context=context)

'''created a simple index page, it’s more like a welcome page for every website or API,
notice the decorator used in line 9, what that decorator does it that it checks the API request method to see if it’s a
GET request being made, if it’s not a GET request then you won’t be able to access that function.'''

@api_view(['GET'])
def howtooperate(request):
    template = loader.get_template('api/HowToOperate.html')
    context = {}
    return render(request, 'api/HowToOperate.html', context=context)


#from .models import RunDB
from .forms import RunForm
from .machinelearning_models import runmodels

def run(request):
    if request.method == "POST":
        #uploaded_file = request.FILES['filename']
        robj = RunForm(request.POST, request.FILES)

    outputs = {}
    #outputs = runmodels(robj.CSV_FILE, robj.DEP_VAR, robj.TASK)
    #'''
    dictt = robj.is_valid()
    if dictt[2]:
        outputs = runmodels(dictt[1], robj.DEP_VAR, robj.TASK)
    else:
        robj = RunForm()
    #'''

    #raise Exception("Accuracies: ",accuracies)
    return  render(request, 'api/Results.html', context={'data':outputs})
