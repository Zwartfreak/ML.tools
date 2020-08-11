from .preprocessing import *

def runmodels(p_datacsv, p_depvars, p_task):
    if p_task == "1":
        return comp_regression(p_datacsv, p_depvars)
    elif p_task == "2":
        return comp_classification(p_datacsv, p_depvars)
    elif p_task == "3":
        return comp_clustering(p_datacsv, p_depvars)
    else:
        raise Exception("There can be no exception. "+"Arguments: ", p_datacsv, p_depvars, p_task)
