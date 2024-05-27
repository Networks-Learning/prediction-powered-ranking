from Result import Result

class ExperimentCollection:
    def __init__(self,input_directory,n,alpha):
        self.experiments=[]
        self.parameter_list=[]
        for nn in n:
            for a in alpha:
                # result=Result(input_directory,{'n':nn,'alpha':a})
                result=Result(input_directory+'n'+str(nn)+'_a'+str(a).split('.')[1]+'/')
                self.experiments.append(result)
                self.parameter_list.append({'n':nn,'alpha':a})

    def do_analysis(self,param_list=[]):
        if param_list==[]:
            param_list=self.parameter_list
        for experiment in self.experiments:
            if experiment.parameters in param_list:
                experiment.do_analysis()