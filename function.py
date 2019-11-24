#  Copyright 2019, Pedro Gomes.
#
#  This file is part of FADO.
#
#  FADO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FADO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with FADO.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# Class to define functions
class Function:
    def __init__(self,name="",outFile="",outParser=None):
        self._name = name

        # where and how the output value is obtained
        self.setOutput(outFile,outParser)
        
        # evaluation pipelines for value and gradient
        self._funEval = []
        self._gradEval = []

        # define inputs, where and how their gradients are obtained
        self._variables = []
        self._gradFiles = []
        self._gradParse = []

    def addInputVariable(self,variable,gradFile,gradParser):
        self._variables.append(variable)
        self._gradFiles.append(gradFile)
        self._gradParse.append(gradParser)

    def getVariables(self):
        return self._variables

    def getParameters(self):
        parameters = []
        for evl in self._funEval:
            parameters += evl.getParameters()
        for evl in self._gradEval:
            parameters += evl.getParameters()
        return parameters

    def setOutput(self,file,parser):
        self._outFile = file
        self._outParser = parser

    def addValueEvalStep(self,evaluation):
        self._funEval.append(evaluation)

    def addGradientEvalStep(self,evaluation):
        self._gradEval.append(evaluation)

    def getValue(self):
        # check if we can retrive the value
        for evl in self._funEval:
            if not evl.isRun():
                self._sequentialEval(self._funEval)
                break
        #end
        return self._outParser.read(self._outFile)

    def getGradient(self,mask=None):
        # check if we can retrive the gradient
        for evl in self._gradEval:
            if not evl.isRun():
                self._sequentialEval(self._gradEval)
                break
        #end

        # determine size of gradient vector
        size = 0
        if mask is None: src = self._variables
        else:            src = mask.keys()
        for var in src:
            size += var.getSize()

        # populate gradient vector
        gradient = np.ndarray((size,))
        idx = 0
        for var,file,parser in zip(self._variables,self._gradFiles,self._gradParse):
            grad = parser.read(file)
            if mask is not None: idx = mask[var]
            try:
                for val in grad:
                    gradient[idx] = val
                    idx += 1
            except:
                gradient[idx] = grad
                idx += 1
            #end
        #end
        
        return gradient
    #end

    def _sequentialEval(self,evals):
        for evl in evals:
            evl.initialize()
            evl.run()
        #end
    #end

    def resetValueEvalChain(self):
        self._resetEvals(self._funEval)

    def resetGradientEvalChain(self):
        self._resetEvals(self._gradEval)

    def _resetEvals(self,evals):
        for evl in evals:
            evl.finalize()
        #end
    #end

    def getValueEvalChain(self):
        return self._funEval

    def getGradientEvalChain(self):
        return self._gradEval
#end
