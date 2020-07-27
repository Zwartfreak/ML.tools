from django import forms
import pandas as pd

noOfCols = df = 0

class RunForm(forms.Form):
    TASK = forms.IntegerField()
    CSV_FILE = forms.FileField()
    DEP_VAR = forms.IntegerField()

    def __init__(self, post, files):
        global df
        self.TASK = post['task']
        self.DEP_VAR = post['dep_var'] # or dep_var = request.POST.get('dep_var')
        self.CSV_FILE = files['filename']
        df = pd.read_csv(self.CSV_FILE)


    def check_task(self):
        if self.TASK == "0":
          raise forms.ValidationError("Please choose a Task.")
        return True #self.TASK

    def check_csv(self):
        try:
            global df, noOfCols
            noOfCols = len(df.columns)
        except Exception as e:
            raise forms.ValidationError("Please enter a file in 'csv' format.",e)

        return True #self.CSV_FILE


    def check_depvar(self):
        global noOfCols
        if self.DEP_VAR == '':
            self.DEP_VAR = -1
            return True
        elif (int)(self.DEP_VAR) >= noOfCols:
            raise forms.ValidationError("Please enter a valid column."+self.DEP_VAR+" "+str(noOfCols))
        else:
            return True #self.DEP_VAR


    def is_valid(self):
        global df
        if self.check_csv() and self.check_depvar() and self.check_task() :
            return {1:df,2:True}
        else:
            return {1:None,2:False}
