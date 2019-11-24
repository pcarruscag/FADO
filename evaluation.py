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

import os
import shutil
import subprocess as sp


# Class to define an execution of an external code
class ExternalRun:
    def __init__(self,dir="",command=""):
        self._dataFiles = []
        self._confFiles = []
        self._workDir = dir
        self._command = command
        self._process = None
        self._parameters = []
        self.finalize()

    def addData(self,file):
        self._dataFiles.append(file)

    def addConfig(self,file):
        self._confFiles.append(file)

    def addParameter(self,param):
        self._parameters.append(param)

    def getParameters(self):
        return self._parameters

    def setWorkDir(self,dir):
        self._workDir = dir

    def setCommand(self,command):
        self._command = command

    def initialize(self,variables):
        if self._isIni: return
        
        os.mkdir(self._workDir)
        for file in self._dataFiles:
            shutil.copy(file,self._workDir)

        for file in self._confFiles:
            shutil.copy(file,self._workDir)
            for par in self._parameters:
                par.writeToFile(os.path.join(self._workDir,file))
            for var in variables:
                var.writeToFile(os.path.join(self._workDir,file))

        self._process = sp.Popen(self._command,cwd=self._workDir,
                        shell=True,stdout=sp.PIPE,stderr=sp.PIPE)

        self._isIni = True
    #end

    def run(self,timeout=None):
        if not self._isIni:
            raise RuntimeError("Run was not initialized.")
        if self._isRun: return self._retcode
        self._retcode = self._process.wait(timeout)
        self._isRun = True
        return self._retcode

    def poll(self):
        if not self._isIni:
            raise RuntimeError("Run was not initialized.")
        if self._isRun: return self._retcode
        retcode = self._process.poll()
        if retcode is not None:
            self._retcode = retcode
            self._isRun = True
        return self._retcode

    def isRun(self):
        return self._isRun

    # reset "lazy" flags
    def finalize(self):
        self._isIni = False
        self._isRun = False
        self._retcode = -100
